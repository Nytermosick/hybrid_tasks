import time
import sys
import threading
from collections import deque
import numpy as np
import pinocchio as pin

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

from utils import compute_gait_phase, compute_gait_stance, quat_rotate_inverse
from hybrid_tasks.assets.robots import DEFAULT_JOINT_POS_NP as DEFAULT_JOINT_POS
from hybrid_tasks.assets.robots import KPj, KDj,\
                                       GAIT_PERIOD, GAIT_OFFSET, GAIT_THRESHOLD,\
                                       COMMAND_STANDING_THRESHOLD, G1_NUM_MOTOR
from hybrid_tasks.assets.robots.unitree_g1.custom_actuator import MATCHING_DICT
from hybrid_tasks.assets.robots.unitree_g1.g1_constants_custom import DEFAULT_JOINT_POS as DEFAULT_JOINT_POS_DICT

class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints

class ObsData:
    def __init__(self, joints_dim=G1_NUM_MOTOR, action_dim=G1_NUM_MOTOR, obs_dim=99, history_len = 1):
        self.obs_dim = obs_dim
        self.history_len = history_len

        self.velocity_commands = np.zeros(3) # Desired velocity commands (x, y, wz)
        self.yaw_orientation_error = np.zeros(1)

        self.joint_pos = np.zeros(joints_dim) # Joint positions for left and right legs
        self.joint_vel = np.zeros(joints_dim) # Joint velocities for left and right legs
 
        self.joint_pos_rel = np.zeros(joints_dim) # Joint positions for left and right legs relative to default values 
        self.joint_vel_rel = np.zeros(joints_dim) # Joint velocities for left and right legs relative to default values

        self.last_action = np.zeros(action_dim)

        self.base_quat = np.zeros(4)  # Base quaternion (w, x, y, z)
        self.base_pos_wf = np.zeros(3)

        self.projected_gravity = np.zeros(3) # Projected gravity in Body Frame
        
        self.base_ang_vel_b = np.zeros(3)   # Angular velocity of body from IMU
        self.base_lin_accel_b = np.zeros(3) # Linear acceleration of body from IMU

        self.gait_phase = np.zeros(2)

        self.shoulders_base_wf = np.zeros(2) # Shoulders of legs relative to body in World Frame

        self.Rwl = np.zeros((3,3))
        self.Rwr = np.zeros((3,3))

        self.J_left = np.zeros((6, 6))  # Jacobian of left leg in World Frame (6x6)
        self.J_right = np.zeros((6, 6)) # Jacobian of right leg in World Frame (6x6)

        self.gravity_torques = np.zeros(joints_dim)
        self.bias_torques = np.zeros(joints_dim + 6)
        self.mass_matrix = np.zeros([joints_dim + 6, joints_dim + 6])

        self.is_stance = np.zeros(2)

        self.time = 0.0

        self.obs_base_ang_vel = deque(maxlen=history_len)
        self.obs_projected_gravity = deque(maxlen=history_len)
        self.obs_velocity_commands = deque(maxlen=history_len)
        self.obs_yaw_orientation_error = deque(maxlen=history_len)
        self.obs_joint_pos_rel = deque(maxlen=history_len)
        self.obs_joint_vel_rel = deque(maxlen=history_len)
        self.obs_last_action = deque(maxlen=history_len)
        self.obs_gait_phase = deque(maxlen=history_len)

        self.obs_full_vector = np.zeros(obs_dim * history_len)

    def init_obs_buffers(self):
        for _ in range(self.history_len):
            self._update_obs_buffers()

    def _update_obs_buffers(self): # TODO: возможно понадобится масштабирование наблюдений
        self.obs_base_ang_vel.append(self.base_ang_vel_b)
        self.obs_projected_gravity.append(self.projected_gravity)
        self.obs_velocity_commands.append(self.velocity_commands)
        self.obs_yaw_orientation_error.append(self.yaw_orientation_error)
        self.obs_joint_pos_rel.append(self.joint_pos_rel)
        self.obs_joint_vel_rel.append(self.joint_vel_rel)
        self.obs_last_action.append(self.last_action)
        self.obs_gait_phase.append(self.gait_phase)

    def get_obs_full_vector(self):
        self._update_obs_buffers()

        base_ang = np.concatenate(list(self.obs_base_ang_vel), axis=0)
        proj_g   = np.concatenate(list(self.obs_projected_gravity), axis=0)
        vel_cmd  = np.concatenate(list(self.obs_velocity_commands), axis=0)
        yaw_err  = np.concatenate(list(self.obs_yaw_orientation_error), axis=0)
        jpos     = np.concatenate(list(self.obs_joint_pos_rel), axis=0)
        jvel     = np.concatenate(list(self.obs_joint_vel_rel), axis=0)
        laction = np.concatenate(list(self.obs_last_action), axis=0)
        gphase = np.concatenate(list(self.obs_gait_phase), axis=0)

        obs_full = np.concatenate([base_ang, proj_g, vel_cmd, jpos, jvel, laction, gphase], axis=0).astype(np.float32)

        return obs_full


class G1_Env:
    def __init__(self, interface, xml_path, control_dt, velocity_commands=[0, 0, 0],
                 action_dim=G1_NUM_MOTOR, obs_dim=99, history_len=1,
                 gait_stop_transition_time=1.0):
        self.control_dt = control_dt

        self.gait_stop_transition_time = gait_stop_transition_time
        self.gait_active_until = 0.0
        self.last_active_gait_command = np.zeros(3)
        
        self.mode_machine = 0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()  
        self.low_state = None 
        self.update_mode_machine = False
        self.crc = CRC()
        self.lock = threading.Lock()

        self.in_default_pos = False
        self.terminated = False

        self.default_torques = np.zeros(G1_NUM_MOTOR)
        self.default_q       = DEFAULT_JOINT_POS.copy()
        self.default_dq      = np.zeros(G1_NUM_MOTOR)
        self._init_motor_characteristic()

        self.pin_model = pin.buildModelFromMJCF(xml_path)

        # foot_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
        # self.foot_frame_ids = [self.pin_model.getFrameId(name) for name in foot_names]
        self.foot_frame_ids = self._ensure_foot_site_frames()

        self.pin_data = self.pin_model.createData()

        self.q_pin_def = np.zeros(G1_NUM_MOTOR+7)
        # self.q_pin_def[2] = -0.793
        self.q_pin_def[6] = 1.0

        self.velocity_commands = velocity_commands

        self.obs_data = ObsData(joints_dim=G1_NUM_MOTOR, obs_dim=obs_dim, history_len=history_len, action_dim=action_dim)

        self.time = 0.0

        self._init_dds(interface)

    def _init_motor_characteristic(self):
        actuator_params = [MATCHING_DICT[name] for name in DEFAULT_JOINT_POS_DICT.keys()]
        self.motor_tau_acc = np.array([params.tau_acc for params in actuator_params], dtype=float)
        self.motor_tau_br = np.array([params.tau_br for params in actuator_params], dtype=float)
        self.motor_v1 = np.array([params.v1 for params in actuator_params], dtype=float)
        self.motor_v2 = np.array([params.v2 for params in actuator_params], dtype=float)

    def _get_current_motor_state(self):
        if self.low_state is None:
            return None, None

        with self.lock:
            motors = self.low_state.motor_state
            q_cur = np.array([motors[i].q for i in range(G1_NUM_MOTOR)], dtype=float)
            dq_cur = np.array([motors[i].dq for i in range(G1_NUM_MOTOR)], dtype=float)
        return q_cur, dq_cur

    def _apply_motor_characteristic(self, *, tau, q, dq, Kp, Kd):
        q_cur, dq_cur = self._get_current_motor_state()
        if q_cur is None:
            return tau

        raw_torques = tau + Kp * (q - q_cur) + Kd * (dq - dq_cur)

        motoring_mask = (dq_cur * raw_torques) > 1e-6
        tau_max_0 = np.where(motoring_mask, self.motor_tau_acc, self.motor_tau_br)

        v_abs = np.abs(dq_cur)
        denom = np.maximum(self.motor_v2 - self.motor_v1, 1e-6)
        linear_scale = 1.0 - (v_abs - self.motor_v1) / denom
        linear_scale = np.clip(linear_scale, 0.0, 1.0)

        tau_limit = np.where(v_abs < self.motor_v1, tau_max_0, tau_max_0 * linear_scale)
        tau_limit = np.where(v_abs > self.motor_v2, 0.0, tau_limit)

        clipped_torques = np.clip(raw_torques, -tau_limit, tau_limit)
        pd_torques = Kp * (q - q_cur) + Kd * (dq - dq_cur)
        return clipped_torques - pd_torques

    def _ensure_foot_site_frames(self):
        foot_sites = {
            "left_foot": ("left_ankle_roll_link", np.array([0.04, 0.0, -0.037])),
            "right_foot": ("right_ankle_roll_link", np.array([0.04, 0.0, -0.037])),
        }

        foot_frame_ids = []
        for site_name, (body_name, site_offset) in foot_sites.items():
            frame_id = self.pin_model.getFrameId(site_name)
            if frame_id < self.pin_model.nframes:
                foot_frame_ids.append(frame_id)
                continue

            body_frame_id = self.pin_model.getFrameId(body_name)
            if body_frame_id >= self.pin_model.nframes:
                raise ValueError(f"Pinocchio model does not contain body frame '{body_name}'")

            body_frame = self.pin_model.frames[body_frame_id]
            site_placement = pin.SE3(np.eye(3), site_offset)
            foot_placement = body_frame.placement * site_placement

            try:
                foot_frame = pin.Frame(
                    site_name,
                    body_frame.parentJoint,
                    body_frame_id,
                    foot_placement,
                    pin.FrameType.OP_FRAME,
                )
            except TypeError:
                foot_frame = pin.Frame(
                    site_name,
                    body_frame.parentJoint,
                    foot_placement,
                    pin.FrameType.OP_FRAME,
                )

            foot_frame_ids.append(self.pin_model.addFrame(foot_frame))
        return foot_frame_ids

    def _init_dds(self, interface):
        # Initialization of DDS message factory
        ChannelFactoryInitialize(1 if interface=="lo" else 0, interface)

        # If we launch on real robot, we need to turn off internal controller of the robot
        if interface != "lo":
            msc = MotionSwitcherClient()
            msc.SetTimeout(5.0)
            msc.Init()

            status, result = msc.CheckMode()
            while result['name']:
                print("Attempt to turn off the internal controller...")
                msc.ReleaseMode()
                status, result = msc.CheckMode()
                time.sleep(1)
            print("The internal controller has been successfully turned off!")

        # create publisher #
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._LowStateHandler, 10)

        print("\nSubcriber and publisher have been successfully created!")

    def check_angles(self, obs_data: ObsData|None = None):
        if obs_data is None:
            obs_data = self.get_measurements()

        w, x, y, z = obs_data.base_quat
        roll = np.arctan2(2*(w*x+y*z), 1-2*(x**2+y**2))
        pitch = np.arcsin(2*(w*y-z*x))
        if (abs(np.rad2deg(roll)) > 45.0 or abs(np.rad2deg(pitch)) > 45.0):
            print("ERROR! Invalid roll/pitch angles for normal work!\nSwitching to damping mode...")
            self.enable_damping_mode()
            raise BaseException("ERROR! Incorrect position of the robot!")

    def move_to_default_pos(self):
        self.check_angles(self.obs_data)
        
        print("\nMoving to default pos.")
        # move time 5s
        total_time = 5
        num_step = int(total_time / self.control_dt)
    
        # record the current pos
        init_joint_pos = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        for i in range(G1_NUM_MOTOR):
            init_joint_pos[i] = self.low_state.motor_state[i].q
        init_joint_pos = init_joint_pos.copy()
        
        # move to default pos
        for i in range(num_step):
            alpha = (i+1) / num_step
            target_pos = init_joint_pos * (1 - alpha) + self.default_q * alpha
            self.send_low_cmd_msg(tau=self.default_torques,
                                  q=target_pos,
                                  dq=self.default_dq)
            time.sleep(self.control_dt)

        print("Robot moved to default pos! Waiting...")
        self.in_default_pos = True

    def stay_in_default_pos(self):
        self.check_angles(self.obs_data)

        self.send_low_cmd_msg(tau=self.default_torques,
                              q=self.default_q,
                              dq=self.default_dq)
        time.sleep(self.control_dt)


    def disable_elastic_band(self):
        self.send_low_cmd_msg(tau=np.zeros(G1_NUM_MOTOR),
                              q = np.zeros(G1_NUM_MOTOR),
                              dq = np.ones(G1_NUM_MOTOR) * 0.001,
                              Kp=np.zeros(G1_NUM_MOTOR),
                              Kd=np.zeros(G1_NUM_MOTOR))
        time.sleep(self.control_dt)

    def _LowStateHandler(self, msg: LowState_):
        with self.lock:
            self.low_state = msg
        # print(self.low_state)

        # Checking for first message
        if self.update_mode_machine == False:
            self.mode_machine = self.low_state.mode_machine
            self.update_mode_machine = True

    def _fill_msg(self, *, tau, q, dq, Kp, Kd):
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine
        for i in range(G1_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].mode = 1 # 1:Enable, 0:Disable
            self.low_cmd.motor_cmd[i].tau = tau[i]
            self.low_cmd.motor_cmd[i].q = q[i]
            self.low_cmd.motor_cmd[i].dq = dq[i]
            self.low_cmd.motor_cmd[i].kp = Kp[i]
            self.low_cmd.motor_cmd[i].kd = Kd[i]

    def enable_damping_mode(self):
        for i in range(G1_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].kd = 1.0
            self.low_cmd.motor_cmd[i].tau = 0.0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

    def send_low_cmd_msg(self, *, tau, q, dq, Kp=KPj, Kd=KDj):
        tau = np.asarray(tau, dtype=float)
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)
        Kp = np.asarray(Kp, dtype=float)
        Kd = np.asarray(Kd, dtype=float)

        tau = self._apply_motor_characteristic(tau=tau, q=q, dq=dq, Kp=Kp, Kd=Kd)

        self._fill_msg(tau=tau, 
                       q=q, 
                       dq=dq,
                       Kp=Kp,
                       Kd=Kd)

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

    def get_measurements(self):
        while self.low_state is None:
            time.sleep(0.0001)

        with self.lock:
            msg = self.low_state

        motors = msg.motor_state
        q_cur = np.array([motors[i].q for i in range(G1_NUM_MOTOR)])
        dq_cur = np.array([motors[i].dq for i in range(G1_NUM_MOTOR)])

        base_quat = np.array(msg.imu_state.quaternion) / np.linalg.norm(msg.imu_state.quaternion)
        projected_gravity = quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0]))

        base_ang_vel_b = np.array(msg.imu_state.gyroscope)
        base_lin_accel_b = np.array(msg.imu_state.accelerometer)

        q_pin = self.q_pin_def.copy()
        q_pin[3:7] = np.array([*base_quat[1:], base_quat[0]])
        q_pin[7:] = q_cur

        pin.forwardKinematics(self.pin_model, self.pin_data, q_pin)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        # Foot site poses in world frame, matching training QP site_names.
        shoulders_base_wf = np.array([self.pin_data.oMf[id].translation for id in self.foot_frame_ids])
        
        Rwl = self.pin_data.oMf[self.foot_frame_ids[0]].rotation
        Rwr = self.pin_data.oMf[self.foot_frame_ids[1]].rotation

        # Foot site Jacobians in World Frame
        J_left = pin.computeFrameJacobian(self.pin_model, self.pin_data, q_pin, self.foot_frame_ids[0], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_right = pin.computeFrameJacobian(self.pin_model, self.pin_data, q_pin, self.foot_frame_ids[1], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        # We need only 6x6 part for each leg
        J_left  = J_left[:, 6:12]
        J_right = J_right[:, 12:18]

        dq_pin = np.zeros(G1_NUM_MOTOR+6)
        dq_pin[3:6] = base_ang_vel_b
        dq_pin[6:] += dq_cur
        bias_torques = pin.rnea(self.pin_model, self.pin_data, q_pin, dq_pin, np.zeros(G1_NUM_MOTOR+6))
        
        mass_matrix = pin.crba(self.pin_model, self.pin_data, q_pin)
        mass_matrix = 0.5 * (mass_matrix + mass_matrix.T)

        self.time += self.control_dt
        command = np.asarray(self.velocity_commands, dtype=float)
        command_norm = np.linalg.norm(command[:2]) + abs(command[2])
        if command_norm > COMMAND_STANDING_THRESHOLD:
            self.gait_active_until = self.time + self.gait_stop_transition_time
            self.last_active_gait_command = command.copy()
        gait_command = self.last_active_gait_command if self.time <= self.gait_active_until else np.zeros_like(command)

        is_stance = compute_gait_stance(
            self.time,
            GAIT_PERIOD,
            GAIT_OFFSET,
            GAIT_THRESHOLD,
            gait_command,
            COMMAND_STANDING_THRESHOLD,
        )

        gait_phase = compute_gait_phase(
            self.time,
            GAIT_PERIOD,
            gait_command,
            COMMAND_STANDING_THRESHOLD,
        )

        cnt = 0.0
        base_height_wf = 0.0
        for i in range(2):
            if is_stance[i]:
                base_height_wf += -shoulders_base_wf[i][2]
                cnt += 1.0
        base_height_wf /= cnt
        base_pos_wf = np.array([0.0, 0.0, base_height_wf])

        # Filling structure
        self.obs_data.velocity_commands = self.velocity_commands
        self.obs_data.joint_pos = q_cur
        self.obs_data.joint_vel = dq_cur
        self.obs_data.joint_pos_rel = q_cur - self.default_q
        self.obs_data.joint_vel_rel = dq_cur - self.default_dq
        self.obs_data.base_quat = base_quat
        self.obs_data.base_pos_wf = base_pos_wf
        self.obs_data.projected_gravity = projected_gravity
        self.obs_data.base_ang_vel_b = base_ang_vel_b
        self.obs_data.base_lin_accel_b = base_lin_accel_b
        self.obs_data.shoulders_base_wf = shoulders_base_wf
        self.obs_data.Rwl = Rwl
        self.obs_data.Rwr = Rwr
        self.obs_data.J_left = J_left
        self.obs_data.J_right = J_right
        self.obs_data.bias_torques = bias_torques
        self.obs_data.mass_matrix = mass_matrix
        self.obs_data.gait_phase = gait_phase
        self.obs_data.is_stance = is_stance
        self.obs_data.time = self.time

        return self.obs_data

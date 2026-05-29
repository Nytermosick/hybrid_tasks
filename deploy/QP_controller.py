import torch
import osqp
from scipy import sparse
import numpy as np
import onnxruntime as ort

import utils as utils
from robot_env import ObsData
from hybrid_tasks.assets.robots import F_MAX_Z, MU,\
                                       BASE_POS_KP, BASE_POS_KD, BASE_ORIENT_KP, BASE_ORIENT_KD,\
                                       BODY_HEIGHT_DESIRED,\
                                       G1_MASS, G1_BASE_INERTIA, QP_YAW_ERROR_LIMIT,\
                                       COMMAND_STANDING_THRESHOLD

from hybrid_tasks.assets.robots import ACTION_SCALE_NP as ACTION_SCALE
from hybrid_tasks.assets.robots import DEFAULT_JOINT_POS_NP as DEFAULT_JOINT_POS

# Shoulders from leg frame to perimeter points
mp = 1.0
mm = 1.0
# mp = 1.25
# mm = 1.11

fcv0 = np.array([+0.12 * mp, +0.030, -0.03])
fcv1 = np.array([+0.12 * mp, -0.030, -0.03])
fcv2 = np.array([-0.05 * mm, +0.025, -0.03])
fcv3 = np.array([-0.05 * mm, -0.025, -0.03])

# Rl_NOM = np.array([-0.026,  0.119, -0.749])  # left foot nominal position in body frame
# Rr_NOM = np.array([-0.026, -0.119, -0.749])  # right foot nominal position in body frame

I3 = np.eye(3)
O3 = np.zeros([3, 3])
O6 = np.zeros((6, 6))
O4X3 = np.zeros([4, 3])
O5X3 = np.zeros([5, 3])
O18X6 = np.zeros([18, 6])


class QPController:
    def __init__(self, policy_path, dt):
        self.session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])

        self.control_dt = dt

        self.Kpl = BASE_POS_KP
        self.Kdl = BASE_POS_KD
        self.Kpa = BASE_ORIENT_KP
        self.Kda = BASE_ORIENT_KD

        self.Q = np.diag([1, 1, 1, 1, 1, 1])

        self.gravity = np.array([0, 0, 9.81, 0, 0, 0])

        self.p_des = np.array([0.0, 0.0, BODY_HEIGHT_DESIRED])
        self.quat_des = np.array([1, 0, 0, 0])
        self.yaw_des_raw = 0.0
        self.yaw_error_limit = QP_YAW_ERROR_LIMIT
        self.linear_acc_des = np.zeros(3)
        self.angular_acc_des = np.zeros(3)

        self.inv_mass = np.eye(3) / G1_MASS
        self.base_inertia = G1_BASE_INERTIA.copy()

        self.need_osqp_init = True
        self.initialized = False

        Af = np.vstack([
            np.array([-1, 0, -MU]),
            np.array([+1, 0, -MU]),
            np.array([0, -1, -MU]),
            np.array([0, +1, -MU]),
        ])

        self.Aff = np.vstack([
            np.hstack([Af, O4X3, O4X3, O4X3]),
            np.hstack([O4X3, Af, O4X3, O4X3]),
            np.hstack([O4X3, O4X3, Af, O4X3]),
            np.hstack([O4X3, O4X3, O4X3, Af]),
        ])

        self.Au_top = np.hstack([self.inv_mass, O3, self.inv_mass, O3])

        self.lb = -np.inf * np.ones(36)
        self.ub =  np.zeros(36)

        self.prev_grf = np.zeros(12)

    def _osqp_setup(self, P, q, A, l, u, verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=1000, polish=False):
        self.prob = osqp.OSQP()
        self.prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=verbose, eps_abs=eps_abs, eps_rel=eps_rel, max_iter=max_iter, polish=polish)

        self.need_osqp_init = False

    def step(self, obs_data: ObsData):
        """
        obs: shape (obs_dim,) or (batch, obs_dim)
        return: actions numpy
        """
        if not self.initialized:
            self.yaw_des_raw = utils.yaw_from_quat(obs_data.base_quat)
            self.quat_des = utils.yaw_quat_from_yaw(self.yaw_des_raw)
            obs_data.yaw_orientation_error = np.zeros(1)
            obs_data.init_obs_buffers()
            self.initialized = True
        else:
            self._update_desired_yaw(obs_data)

        obs = obs_data.get_obs_full_vector()

        if obs.ndim == 1:
            obs = obs[None, :]

        obs = obs.astype(np.float32)
        action = self.session.run(None, {"obs": obs})[0]
        action = np.asarray(action).reshape(-1)
        obs_data.last_action = action.copy()

        # self.desired_linear_acceleration =  Rwbz @ action[29:32] #* 0.5
        # self.desired_angular_acceleration = Rwbz @ action[32:35] #* 0.5

        grf = self.solveQP(obs_data)

        J = np.vstack([np.hstack([obs_data.J_left,  O6]),
                       np.hstack([O6, obs_data.J_right])])
        
        leg_torques = -J.T @ grf

        scaled_actions = action * ACTION_SCALE

        full_torques = obs_data.bias_torques[6:].copy()
        full_torques[:12] += leg_torques

        full_q = DEFAULT_JOINT_POS + scaled_actions
        full_dq = np.zeros_like(full_q)

        return full_torques, full_q, full_dq
    
    def solveQP(self, obs_data: ObsData):

        base_quat_cur = obs_data.base_quat
        Rwb = utils.quat_to_R_wxyz(base_quat_cur)
        base_quat_z_cur = utils.yaw_quat_from_quat(base_quat_cur)
        Rwbz = utils.quat_to_R_wxyz(base_quat_z_cur)

        # print(f"Base yaw: {utils.yaw_from_quat(base_quat_cur):.3f} rad, Desired yaw: {utils.yaw_from_quat(self.quat_des):.3f} rad")

        base_pos_cur = obs_data.base_pos_wf
        base_lin_vel_cur = np.zeros(3)
        base_ang_vel_cur = utils.quat_apply(base_quat_cur, obs_data.base_ang_vel_b)

        base_lin_vel_des = Rwbz @ np.array([obs_data.velocity_commands[0], obs_data.velocity_commands[1], 0.0])

        yaw_rate_cmd = obs_data.velocity_commands[2]
        total_command = np.linalg.norm(obs_data.velocity_commands[:2]) + abs(yaw_rate_cmd)
        yaw_rate_cmd = yaw_rate_cmd if total_command > COMMAND_STANDING_THRESHOLD else 0.0

        base_ang_vel_des = np.array([0.0, 0.0, yaw_rate_cmd])

        lin_pos_error = self.p_des - base_pos_cur
        lin_vel_error = base_lin_vel_des - base_lin_vel_cur
        desired_lin_acc = (self.Kpl * lin_pos_error +
                           self.Kdl * lin_vel_error +
                           self.linear_acc_des)

        ang_pos_error = utils.quat_error(self.quat_des, base_quat_cur)
        ang_vel_error = base_ang_vel_des - base_ang_vel_cur
        desired_ang_acc = (self.Kpa * ang_pos_error +
                           self.Kda * ang_vel_error +
                           self.angular_acc_des)

        desired_acc = np.concatenate((desired_lin_acc, desired_ang_acc))

        mass_matrix = obs_data.mass_matrix
        bias_torques = obs_data.bias_torques

        acc_bias = np.linalg.solve(mass_matrix[:6, :6], bias_torques[:6])
        acc_bias[3:] = utils.quat_apply(base_quat_cur, acc_bias[3:])

        a = desired_acc + acc_bias

        # Ib = self.base_inertia
        Ib = mass_matrix[3:6, 3:6]
        Iw = Rwb @ Ib @ Rwb.T
        Iw_inv = np.linalg.inv(Iw)

        shoulders_wf = obs_data.shoulders_base_wf
        Rwl = obs_data.Rwl
        Rwr = obs_data.Rwr

        Au_bot = np.hstack([Iw_inv @ utils.skew(shoulders_wf[0]),
                            Iw_inv,
                            Iw_inv @ utils.skew(shoulders_wf[1]),
                            Iw_inv])
        Au = np.vstack([self.Au_top, Au_bot])

        q = (-a.T @ self.Q @ Au)
        P =  Au.T @ self.Q @ Au + 1e-4 * np.eye(12)

        B = np.vstack([
            np.hstack([I3, I3, I3, I3]),
            np.hstack([utils.skew(Rwl @ fcv0),
                       utils.skew(Rwl @ fcv1),
                       utils.skew(Rwl @ fcv2),
                       utils.skew(Rwl @ fcv3)]),
        ])
        Bl_inv = np.linalg.pinv(B)

        B = np.vstack([
            np.hstack([I3, I3, I3, I3]),
            np.hstack([utils.skew(Rwr @ fcv0),
                       utils.skew(Rwr @ fcv1),
                       utils.skew(Rwr @ fcv2),
                       utils.skew(Rwr @ fcv3)]),
        ])
        Br_inv = np.linalg.pinv(B)

        Dl = self.Aff @ Bl_inv
        Dr = self.Aff @ Br_inv

        Acwcl = np.vstack([
            Dl,
            np.array([0, 0, +1, 0, 0, 0]),
            np.array([0, 0, -1, 0, 0, 0]),
        ])

        Acwcr = np.vstack([
            Dr,
            np.array([0, 0, +1, 0, 0, 0]),
            np.array([0, 0, -1, 0, 0, 0]),
        ])

        A = np.vstack([
            np.hstack([Acwcl, O18X6]),
            np.hstack([O18X6, Acwcr]),
        ])

        self.ub[16] = F_MAX_Z if obs_data.is_stance[0] else 0
        self.ub[34] = F_MAX_Z if obs_data.is_stance[1] else 0

        P_ut = np.triu(P)
        tiny = 1e-16
        r, c = np.triu_indices(P_ut.shape[0])
        data = P_ut[r, c].copy()
        data[data == 0.0] = tiny

        P = sparse.csc_matrix((data, (r, c)), shape=P_ut.shape)
        A = sparse.csc_matrix(A)

        if self.need_osqp_init:
            self._osqp_setup(P, q, A, self.lb, self.ub)
        else:
            self.prob.update(Px=P.data, q=q, Ax=A.data, l=self.lb, u=self.ub)
            self.prob.warm_start(x=self.prev_grf)

        res = self.prob.solve()

        grf = res.x.astype(float)
        self.prev_grf = grf

        # solved_acc = (Au @ grf) - self.gravity
        # da = solved_acc - desired_acc

        # print(f"{solved_acc=}")
        # print(f"{desired_acc=}")
        # print(f"{da=}")
        
        return grf
    
    def _update_desired_yaw(self, obs_data: ObsData):
        current_yaw = utils.yaw_from_quat(obs_data.base_quat)
        total_command = np.linalg.norm(obs_data.velocity_commands[:2]) + abs(obs_data.velocity_commands[2])
        yaw_rate_des = obs_data.velocity_commands[2] if total_command > COMMAND_STANDING_THRESHOLD else 0.0
        self.yaw_des_raw = utils.wrap_to_pi(
            self.yaw_des_raw + yaw_rate_des * self.control_dt
        )
        yaw_error = utils.wrap_to_pi(self.yaw_des_raw - current_yaw)
        bounded_yaw_error = np.clip(
            yaw_error,
            -self.yaw_error_limit,
            self.yaw_error_limit,
        )
        self.quat_des = utils.yaw_quat_from_yaw(utils.wrap_to_pi(current_yaw + bounded_yaw_error))
        obs_data.yaw_orientation_error = np.array([yaw_error], dtype=float)

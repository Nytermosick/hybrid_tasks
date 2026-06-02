import numpy as np
import onnxruntime as ort
from robot_env import ObsData
import utils as utils
from hybrid_tasks.assets.robots import DEFAULT_JOINT_POS_NP as DEFAULT_JOINT_POS
from hybrid_tasks.assets.robots import ACTION_SCALE_NP as ACTION_SCALE
from hybrid_tasks.assets.robots import COMMAND_STANDING_THRESHOLD

class SimpleController:
    def __init__(self, policy_path, dt):
        self.session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])

        self.control_dt = dt
        self.yaw_des_raw = 0.0
        self.initialized = False
        self.action_dim = self._infer_policy_action_dim()

    def step(self, obs_data: ObsData):
        """
        obs: shape (obs_dim,) or (batch, obs_dim)
        return: actions numpy
        """
        if not self.initialized:
            self.yaw_des_raw = utils.yaw_from_quat(obs_data.base_quat)
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

        if action.ndim == 2 and action.shape[0] == 1:
            action = action[0]

        obs_data.last_action = action

        scaled_actions = action * ACTION_SCALE

        full_torques = np.zeros_like(DEFAULT_JOINT_POS)
        full_q = DEFAULT_JOINT_POS + scaled_actions
        full_dq = np.zeros_like(DEFAULT_JOINT_POS)

        return full_torques, full_q, full_dq

    def _update_desired_yaw(self, obs_data: ObsData):
        current_yaw = utils.yaw_from_quat(obs_data.base_quat)
        total_command = np.linalg.norm(obs_data.velocity_commands[:2]) + abs(obs_data.velocity_commands[2])
        yaw_rate_des = obs_data.velocity_commands[2] if total_command > COMMAND_STANDING_THRESHOLD else 0.0
        self.yaw_des_raw = utils.wrap_to_pi(
            self.yaw_des_raw + yaw_rate_des * self.control_dt
        )
        yaw_error = utils.wrap_to_pi(self.yaw_des_raw - current_yaw)
        obs_data.yaw_orientation_error = np.array([yaw_error], dtype=float)

    def _infer_policy_action_dim(self) -> int:
        output_shape = self.session.get_outputs()[0].shape
        output_dim = output_shape[-1]
        return output_dim

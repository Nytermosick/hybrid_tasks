import numpy as np
import onnxruntime as ort
from robot_env import ObsData
from hybrid_tasks.assets.robots import G1_ACTION_SCALE_CUSTOM as ACTION_SCALE
from hybrid_tasks.assets.robots import DEFAULT_JOINT_POS_NP as DEFAULT_JOINT_POS
from hybrid_tasks.assets.robots import ACTION_SCALE_NP as ACTION_SCALE

class SimpleController:
    def __init__(self, policy_path, dt=None): # dt doesn't use
        self.session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])

        self.initialized = False

    def step(self, obs_data: ObsData):
        """
        obs: shape (obs_dim,) or (batch, obs_dim)
        return: actions numpy
        """
        if not self.initialized:
            obs_data.init_obs_buffers()
            self.initialized = True

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

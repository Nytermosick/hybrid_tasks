"""Joint position action with QP feedforward torques."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import mujoco_warp as mjw
import warp as wp

from mjlab.entity import Entity
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from hybrid_tasks.tasks.velocity import qp

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


ScaleCfg = float | dict[str, float]


@dataclass(kw_only=True)
class ResidualPositionsAndTorquesCfg(JointPositionActionCfg):
  """Residual joint-position action plus feedforward torques from QP.

  This inherits the standard ``JointPositionAction`` behavior:
  The policy action dimension is unchanged. Joint position targets come from the
  policy. When ``use_qp_torques`` is true, feedforward torques are computed
  internally from QP contact wrenches and foot Jacobians.
  """

  use_qp_torques: bool = True

  def build(self, env: ManagerBasedRlEnv) -> ResidualPositionsAndTorques:
    return ResidualPositionsAndTorques(self, env)


class ResidualPositionsAndTorques(JointPositionAction):
  """Write both joint position targets and external feedforward torques."""

  cfg: ResidualPositionsAndTorquesCfg

  def __init__(self, cfg: ResidualPositionsAndTorquesCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    self._effort_targets = torch.zeros_like(self._raw_actions)
    self._zero_velocity_targets = torch.zeros_like(self._raw_actions)

    self._foot_names = ("left_foot", "right_foot")
    self._asset_cfg = SceneEntityCfg("robot", site_names=self._foot_names)
    self._asset_cfg.resolve(self._env.scene)
    self._asset: Entity = self._env.scene[self._asset_cfg.name]

    self._nv = self._asset.data.model.nv
    self._wp_data = self._env.sim.wp_data
    self._wp_model = self._env.sim.wp_model

    self._foot_site_ids = self._asset_cfg.site_ids
    foot_frame_ids = self._asset.indexing.site_ids[self._foot_site_ids]
    self._foot_body_ids = (
      self._asset.data.model.site_bodyid[foot_frame_ids].cpu().numpy()
    )
    self._target_dof_ids = self._asset.indexing.joint_v_adr[
      self._target_ids
    ].long()

    leg_joint_mask = [
        "hip" in name or "knee" in name or "ankle" in name
        for name in self._target_names
    ]
    left_leg_joint_mask = [
      name.startswith("left_") and ("hip" in name or "knee" in name or "ankle" in name)
      for name in self._target_names
    ]
    right_leg_joint_mask = [
      name.startswith("right_") and ("hip" in name or "knee" in name or "ankle" in name)
      for name in self._target_names
    ]
    
    self._left_leg_target_mask = torch.tensor(left_leg_joint_mask, device=self.device)
    self._right_leg_target_mask = torch.tensor(right_leg_joint_mask, device=self.device)
    self._leg_target_mask = self._left_leg_target_mask | self._right_leg_target_mask

    self._qpcfg = qp.QPCfg(num_envs=self.num_envs, device=self.device)

  @property
  def effort_target(self) -> torch.Tensor:
    return self._effort_targets

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    if self.cfg.use_qp_torques:
      self._effort_targets[:] = self._compute_qp_torques()
    else:
      self._effort_targets[:] = 0.0

  def apply_actions(self) -> None:
    super().apply_actions()
    self._entity.set_joint_velocity_target(
      self._zero_velocity_targets, joint_ids=self._target_ids
    )
    self._entity.set_joint_effort_target(
      self._effort_targets, joint_ids=self._target_ids
    )

  def _compute_qp_torques(self) -> torch.Tensor:
    foot_wrenches = qp.solveQP(self._env, self._qpcfg)
    foot_pos_w = self._asset.data.site_pos_w[:, self._foot_site_ids, :].cpu().numpy()
    torque_targets = torch.zeros_like(self._effort_targets)

    for foot_id, body_id in enumerate(self._foot_body_ids):
      with wp.ScopedDevice(self._env.sim.wp_device):
        point_wp = wp.array(foot_pos_w[:, foot_id, :], dtype=wp.vec3)
        body_wp = wp.array(
          np.full((self.num_envs,), body_id, dtype=np.int32), dtype=wp.int32
        )

        jacp_wp = wp.zeros((self.num_envs, 3, self._nv), dtype=float)
        jacr_wp = wp.zeros((self.num_envs, 3, self._nv), dtype=float)
        mjw.jac(self._wp_model, self._wp_data, jacp_wp, jacr_wp, point_wp, body_wp)

      jacp_all = wp.to_torch(jacp_wp)
      jacr_all = wp.to_torch(jacr_wp)
      target_dof_ids = self._target_dof_ids.to(jacp_all.device)
      jacp = jacp_all[:, :, target_dof_ids].to(self.device)
      jacr = jacr_all[:, :, target_dof_ids].to(self.device)

      force = foot_wrenches[foot_id][:, 0:3, :]
      moment = foot_wrenches[foot_id][:, 3:6, :]
      torque_targets += -torch.bmm(jacp.transpose(1, 2), force).squeeze(-1)
      torque_targets += -torch.bmm(jacr.transpose(1, 2), moment).squeeze(-1)
      
    torque_targets[:, ~self._leg_target_mask] = 0.0
    torques = torque_targets + self._masked_bias_torques()
    return torques # TODO: Кориолисовы силы могут быть чувствительными к скоростям суставов

  def _bias_torques(self) -> torch.Tensor:
    qfrc_bias = self._asset.data.data.qfrc_bias
    target_dof_ids = self._target_dof_ids.to(qfrc_bias.device)
    return qfrc_bias[:, target_dof_ids].to(self.device)
  
  def _masked_bias_torques(self) -> torch.Tensor:
    bias_torques = self._bias_torques()
    command = self._env.command_manager.get_command("twist")
    assert command is not None
    is_stance = qp.gaitStanceWithCommand(
      self._env,
      command,
      period=qp.GAIT_PERIOD,
      offset=qp.GAIT_OFFSET,
      threshold=qp.GAIT_THRESHOLD,
      command_threshold=qp.COMMAND_STANDING_THRESHOLD,
    )
    bias_mask = torch.ones_like(bias_torques, dtype=torch.bool)
    left_swing = ~is_stance[:, 0].unsqueeze(1)
    right_swing = ~is_stance[:, 1].unsqueeze(1)
    bias_mask[:, self._left_leg_target_mask] = left_swing.expand(
      -1, int(self._left_leg_target_mask.sum().item())
    )
    bias_mask[:, self._right_leg_target_mask] = right_swing.expand(
      -1, int(self._right_leg_target_mask.sum().item())
    )
    return torch.where(bias_mask, bias_torques, torch.zeros_like(bias_torques))

  def set_feedforward_torques(
    self,
    torques: torch.Tensor,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    """
    Set feedforward joint torques from a QP.
    """
    if env_ids is None:
      env_ids = slice(None)
    self._effort_targets[env_ids] = torques

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    super().reset(env_ids=env_ids)
    self._effort_targets[env_ids] = 0.0
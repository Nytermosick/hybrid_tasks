"""Joint position action with an external feedforward torque channel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


ScaleCfg = float | dict[str, float]


@dataclass(kw_only=True)
class ResidualPositionsAndTorquesCfg(JointPositionActionCfg):
  """Residual joint-position action plus feedforward torques from outside policy.

  This inherits the standard ``JointPositionAction`` behavior:
  The policy action dimension is unchanged. Feedforward torques are expected to
  come from QP, via
  "ResidualPositionsAndTorques.set_feedforward_torques()".
  """

  def build(self, env: ManagerBasedRlEnv) -> ResidualPositionsAndTorques:
    return ResidualPositionsAndTorques(self, env)


class ResidualPositionsAndTorques(JointPositionAction):
  """Write both joint position targets and external feedforward torques."""

  cfg: ResidualPositionsAndTorquesCfg

  def __init__(self, cfg: ResidualPositionsAndTorquesCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    self._effort_targets = torch.zeros_like(self._raw_actions)

  @property
  def effort_target(self) -> torch.Tensor:
    return self._effort_targets

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    self.clear_feedforward_torques()

  def apply_actions(self) -> None:
    super().apply_actions()
    self._entity.set_joint_effort_target(
      self._effort_targets, joint_ids=self._target_ids
    )

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

  def clear_feedforward_torques(
    self, env_ids: torch.Tensor | slice | None = None
  ) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._effort_targets[env_ids] = 0.0

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    super().reset(env_ids=env_ids)
    self.clear_feedforward_torques(env_ids=env_ids)

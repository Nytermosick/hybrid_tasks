from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
import mjlab.utils.lab_api.math as math_utils

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
  return torch.atan2(torch.sin(angle), torch.cos(angle))


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quat.unbind(dim=-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def yaw_orientation_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  error_limit: float | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Yaw-only error between the integrated desired heading and current base heading."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  current_yaw = _yaw_from_quat(math_utils.yaw_quat(asset.data.root_link_quat_w))
  if not hasattr(env, "_desired_yaw_obs"):
    env._desired_yaw_obs = current_yaw.clone()
  if not hasattr(env, "_desired_yaw_obs_last_step"):
    env._desired_yaw_obs_last_step = torch.full_like(env.episode_length_buf, -1)

  reset = env.episode_length_buf == 0
  update = env.episode_length_buf != env._desired_yaw_obs_last_step
  updated_desired_yaw = torch.where(
    reset,
    current_yaw,
    _wrap_to_pi(env._desired_yaw_obs + command[:, 2] * env.step_dt),
  )
  desired_yaw = torch.where(update, updated_desired_yaw, env._desired_yaw_obs)
  env._desired_yaw_obs = desired_yaw.detach()
  env._desired_yaw_obs_last_step = torch.where(
    update,
    env.episode_length_buf,
    env._desired_yaw_obs_last_step,
  )

  yaw_error = _wrap_to_pi(desired_yaw - current_yaw)
  if error_limit is not None:
    yaw_error = torch.clamp(yaw_error, min=-error_limit, max=error_limit)
  return yaw_error.unsqueeze(1)


def phase(env: ManagerBasedRlEnv, period: float, command_name: str, command_threshold: float = 0.1) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase

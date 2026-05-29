"""Unitree G1 vanilla flat walking environment configuration."""

from hybrid_tasks.assets.robots import (
  G1_ACTION_SCALE_CUSTOM,
  GAIT_PERIOD, GAIT_OFFSET, GAIT_THRESHOLD, COMMAND_STANDING_THRESHOLD,
  FOOT_CLEARANCE,
  get_g1_robot_cfg_custom,
)
from hybrid_tasks.tasks.velocity.vanlilla_walk_flat import (
  make_vanilla_walk_flat_env_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from hybrid_tasks.tasks.velocity.custom_residual_action import (
  ResidualPositionsAndTorquesCfg,
)


def g1_vanilla_walk_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 vanilla flat walking configuration."""
  cfg = make_vanilla_walk_flat_env_cfg()

  cfg.scene.entities = {"robot": get_g1_robot_cfg_custom()}

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE_CUSTOM

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names
  cfg.observations["actor"].terms["gait_phase"].params["command_threshold"] = COMMAND_STANDING_THRESHOLD
  cfg.observations["actor"].terms["yaw_orientation_error"].params["command_threshold"] = COMMAND_STANDING_THRESHOLD

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.25,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.rewards["foot_gait"].params["period"] = GAIT_PERIOD
  cfg.rewards["foot_gait"].params["offset"] = GAIT_OFFSET
  cfg.rewards["foot_gait"].params["threshold"] = GAIT_THRESHOLD
  cfg.rewards["foot_gait"].params["command_threshold"] = COMMAND_STANDING_THRESHOLD
  cfg.rewards["yaw_orientation_error_l2"].params["command_threshold"] = COMMAND_STANDING_THRESHOLD

  cfg.rewards["foot_clearance"].params["target_height"] = FOOT_CLEARANCE
  
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}

    twist_cmd.ranges.lin_vel_x = (-1.0, 2.0)
    twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
    twist_cmd.ranges.ang_vel_z = (-1.5, 1.5)

  return cfg


def g1_qp_without_acc_walk_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat walking configuration with QP torques without acceleration in the policy."""
  cfg = g1_vanilla_walk_flat_env_cfg(play=play)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, ResidualPositionsAndTorquesCfg)
  joint_pos_action.use_qp_torques = True

  if play:
    cfg.commands["twist"].ranges.lin_vel_x = (-1.1, 1.1)
    cfg.commands["twist"].ranges.lin_vel_y = (-0.3, 0.3)
    cfg.commands["twist"].ranges.ang_vel_z = (-1.5, 1.5)
    # cfg.commands["twist"].ranges.lin_vel_x = (0.0, 0.0)
    # cfg.commands["twist"].ranges.lin_vel_y = (0.0, 0.0)
    # cfg.commands["twist"].ranges.ang_vel_z = (0.0, 0.0)

    # cfg.events["reset_base"].params["pose_range"]["x"] = (-0.0, 0.0)
    # cfg.events["reset_base"].params["pose_range"]["y"] = (-0.0, 0.0)
    # cfg.events["reset_base"].params["pose_range"]["z"] = (-0.0, 0.0)
    # cfg.events["reset_base"].params["pose_range"]["roll"] = (-0.0, 0.0)
    # cfg.events["reset_base"].params["pose_range"]["pitch"] = (-0.0, 0.0)
    # cfg.events["reset_base"].params["pose_range"]["yaw"] = (-0.0, 0.0)

    # cfg.events["reset_robot_joints"].params["position_range"] = (-0.0, 0.0)
    # cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.0, 0.0)


  return cfg

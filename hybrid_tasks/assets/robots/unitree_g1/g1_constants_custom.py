"""Unitree G1 constants."""

from pathlib import Path

import mujoco
import numpy as np

from hybrid_tasks import SRC_PATH
from hybrid_tasks.assets.robots.unitree_g1.custom_actuator import XmlCustomActuatorCfg, CustomActuator, MATCHING_DICT
from mjlab.actuator import ActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

G1_NUM_MOTOR = 29

CONTROL_DT = 0.01

GAIT_PERIOD = 0.8
GAIT_OFFSET = [0, 0.5]
GAIT_THRESHOLD = 0.56  # phase < GAIT_THRESHOLD is stance (Stance part)
STANCE_PERIOD = GAIT_THRESHOLD * GAIT_PERIOD

F_MAX_Z = 600
MU = 1.0

BASE_POS_KP    = np.array([0, 0, 1]) * 40
BASE_POS_KD    = np.array([1, 1, 1]) * 7
BASE_ORIENT_KP = np.array([1, 1, 1]) * 250
BASE_ORIENT_KD = np.array([1, 1, 1]) * 7

FOOT_CLEARANCE = 0.10
COM_HEIGHT_DESIRED = 0.65

##
# MJCF and assets.
##

G1_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_g1" / "xmls" / "g1_with_motors.xml"
)
assert G1_XML.exists()

def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    for actuator in spec.actuators:
        joint_name = actuator.target
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
        actuator.gainprm[0] = 1

        limit = MATCHING_DICT[joint_name].effort_limit
        actuator.forcelimited = True
        actuator.forcerange = (-limit, limit)
    return spec


actuators: list[ActuatorCfg] = []
for expression in MATCHING_DICT.keys():
    actuator = XmlCustomActuatorCfg(target_names_expr=(expression,))
    actuators.append(actuator)

##
# Final config.
##
G1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=tuple(actuators),
  soft_joint_pos_limit_factor=0.9,
)

##
# Keyframe config.
##

DEFAULT_JOINT_POS = {
  "left_hip_pitch_joint"      : -0.312,
  "left_hip_roll_joint"       :  0.0,
  "left_hip_yaw_joint"        :  0.0,
  "left_knee_joint"           :  0.669,
  "left_ankle_pitch_joint"    : -0.363,
  "left_ankle_roll_joint"     :  0.0,

  "right_hip_pitch_joint"     : -0.312,
  "right_hip_roll_joint"      :  0.0,
  "right_hip_yaw_joint"       :  0.0,
  "right_knee_joint"          :  0.669,
  "right_ankle_pitch_joint"   : -0.363,
  "right_ankle_roll_joint"    :  0.0,

  "waist_yaw_joint"           :  0.0,
  "waist_roll_joint"          :  0.0,
  "waist_pitch_joint"         :  0.0,

  "left_shoulder_pitch_joint" :  0.2,
  "left_shoulder_roll_joint"  :  0.2,
  "left_shoulder_yaw_joint"   :  0.2,
  "left_elbow_joint"          :  0.6,
  "left_wrist_roll_joint"     :  0.0,
  "left_wrist_pitch_joint"    :  0.0,
  "left_wrist_yaw_joint"      :  0.0,

  "right_shoulder_pitch_joint":  0.2,
  "right_shoulder_roll_joint" : -0.2,
  "right_shoulder_yaw_joint"  :  0.2,
  "right_elbow_joint"         :  0.6,
  "right_wrist_roll_joint"    :  0.0,
  "right_wrist_pitch_joint"   :  0.0,
  "right_wrist_yaw_joint"     :  0.0,
}

DEFAULT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.76),
  joint_pos=DEFAULT_JOINT_POS,
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

def get_g1_robot_cfg_custom() -> EntityCfg:
  """Get a fresh G1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=DEFAULT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_ARTICULATION,
  )


G1_ACTION_SCALE_CUSTOM: dict[str, float] = {}
for actuator_name, actuator_type in MATCHING_DICT.items():
  assert isinstance(actuator_type, CustomActuator)
  e = actuator_type.effort_limit
  s = actuator_type.stiffness
  assert e is not None
  G1_ACTION_SCALE_CUSTOM[actuator_name] = 0.25 * e / s

KPj = np.array([value.stiffness for value in MATCHING_DICT.values()])
KDj = np.array([value.damping for value in MATCHING_DICT.values()])

DEFAULT_JOINT_POS_NP = np.array(list(DEFAULT_JOINT_POS.values()), dtype=float)
ACTION_SCALE_NP = np.array(list(G1_ACTION_SCALE_CUSTOM.values()), dtype=float)

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_robot_cfg_custom())

  viewer.launch(robot.spec.compile())

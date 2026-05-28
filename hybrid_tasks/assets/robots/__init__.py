from .unitree_g1.g1_constants import (
  G1_ACTION_SCALE as G1_ACTION_SCALE,
)
from .unitree_g1.g1_constants import (
  get_g1_robot_cfg as get_g1_robot_cfg,
)

from .unitree_g1.g1_constants_custom import (
  FOOT_CLEARANCE,
  G1_ACTION_SCALE_CUSTOM as G1_ACTION_SCALE_CUSTOM,
)
from .unitree_g1.g1_constants_custom import DEFAULT_JOINT_POS_NP, KPj, KDj,\
                                            ACTION_SCALE_NP,\
                                            GAIT_PERIOD, GAIT_OFFSET, GAIT_THRESHOLD, STANCE_PERIOD,\
                                            FOOT_CLEARANCE,\
                                            CONTROL_DT, G1_NUM_MOTOR,\
                                            BODY_HEIGHT_DESIRED, \
                                            G1_MASS, G1_BASE_INERTIA, \
                                            F_MAX_Z, MU,\
                                            BASE_POS_KP, BASE_POS_KD, BASE_ORIENT_KP, BASE_ORIENT_KD,\
                                            COMMAND_STANDING_THRESHOLD, QP_YAW_ERROR_LIMIT
from .unitree_g1.g1_constants_custom import (
  get_g1_robot_cfg_custom as get_g1_robot_cfg_custom,
)

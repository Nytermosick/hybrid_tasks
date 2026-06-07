from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from mjlab.actuator import ActuatorCmd, XmlActuator, XmlActuatorCfg
from mjlab.utils.actuator import reflected_inertia_from_two_stage_planetary
import torch

if TYPE_CHECKING:
  from mjlab.entity import Entity

@dataclass
class CustomActuator:
    tau_acc:      float
    tau_br:       float      
    v1:           float       
    v2:           float
    effort_limit: float 
    stiffness:    float 
    damping:      float 
    fs:           float = 0.0
    fd:           float = 0.0
    va:           float = 0.01

# Motor specs (from Unitree).
ROTOR_INERTIAS_5020 = (
  0.139e-4,
  0.017e-4,
  0.169e-4,
)
GEARS_5020 = (
  1,
  1 + (46 / 18),
  1 + (56 / 16),
)
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_5020, GEARS_5020
)

ROTOR_INERTIAS_7520_14 = (
  0.489e-4,
  0.098e-4,
  0.533e-4,
)
GEARS_7520_14 = (
  1,
  4.5,
  1 + (48 / 22),
)
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_14, GEARS_7520_14
)

ROTOR_INERTIAS_7520_22 = (
  0.489e-4,
  0.109e-4,
  0.738e-4,
)
GEARS_7520_22 = (
  1,
  4.5,
  5,
)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_22, GEARS_7520_22
)

ROTOR_INERTIAS_4010 = (
  0.068e-4,
  0.0,
  0.0,
)
GEARS_4010 = (
  1,
  5,
  5,
)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_4010, GEARS_4010
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

ACTUATOR_5020 = CustomActuator(
    tau_acc = 24.8,
    tau_br = 31.9,
    v1 = 30.86,
    v2 = 40.13,
    effort_limit = 25.0,
    stiffness = STIFFNESS_5020,
    damping = DAMPING_5020,
    fs = 0.6,
    fd = 0.06
)

ACTUATOR_7520_14 = CustomActuator(
    tau_acc = 71.0,
    tau_br = 83.3,
    v1 = 22.63,
    v2 = 35.52,
    effort_limit = 88.0,
    stiffness = STIFFNESS_7520_14,
    damping = DAMPING_7520_14,
    fs = 1.6,
    fd = 0.16
)

ACTUATOR_7520_22 = CustomActuator(
    tau_acc = 111.0,
    tau_br = 131.0,
    v1 = 14.50,
    v2 = 22.70,
    effort_limit = 139.0,
    stiffness = STIFFNESS_7520_22,
    damping = DAMPING_7520_22,
    fs = 2.4,
    fd = 0.24
)

ACTUATOR_4010 = CustomActuator(
    tau_acc = 4.8,
    tau_br = 8.6,
    v1 = 15.30,
    v2 = 24.76,
    effort_limit = 5.0,
    stiffness = STIFFNESS_4010,
    damping = DAMPING_4010,
    fs = 0.6,
    fd = 0.06
)

ACTUATOR_WAIST = CustomActuator(
  tau_acc = ACTUATOR_5020.tau_acc * 2,
  tau_br = ACTUATOR_5020.tau_br * 2,
  v1 = ACTUATOR_5020.v1,
  v2 = ACTUATOR_5020.v2,
  effort_limit = ACTUATOR_5020.effort_limit * 2,
  stiffness = STIFFNESS_5020 * 2,
  damping = DAMPING_5020 * 2,
  fs = ACTUATOR_5020.fs * 2,
  fd = ACTUATOR_5020.fd * 2,
  va = ACTUATOR_5020.va
)
ACTUATOR_ANKLE = CustomActuator(
  tau_acc = ACTUATOR_5020.tau_acc * 2,
  tau_br = ACTUATOR_5020.tau_br * 2,
  v1 = ACTUATOR_5020.v1,
  v2 = ACTUATOR_5020.v2,
  effort_limit = ACTUATOR_5020.effort_limit * 2,
  stiffness = STIFFNESS_5020 * 2,
  damping = DAMPING_5020 * 2,
  fs = ACTUATOR_5020.fs * 2,
  fd = ACTUATOR_5020.fd * 2,
  va = ACTUATOR_5020.va
)

MATCHING_DICT = {
    "left_hip_pitch_joint"       : ACTUATOR_7520_22,
    "left_hip_roll_joint"        : ACTUATOR_7520_22,
    "left_hip_yaw_joint"         : ACTUATOR_7520_14,
    "left_knee_joint"            : ACTUATOR_7520_22,
    "left_ankle_pitch_joint"     : ACTUATOR_ANKLE,
    "left_ankle_roll_joint"      : ACTUATOR_ANKLE,

    "right_hip_pitch_joint"      : ACTUATOR_7520_22,
    "right_hip_roll_joint"       : ACTUATOR_7520_22,
    "right_hip_yaw_joint"        : ACTUATOR_7520_14,
    "right_knee_joint"           : ACTUATOR_7520_22,
    "right_ankle_pitch_joint"    : ACTUATOR_ANKLE,
    "right_ankle_roll_joint"     : ACTUATOR_ANKLE,

    "waist_yaw_joint"            : ACTUATOR_7520_14,
    "waist_roll_joint"           : ACTUATOR_WAIST,
    "waist_pitch_joint"          : ACTUATOR_WAIST,

    "left_shoulder_pitch_joint"  : ACTUATOR_5020,
    "left_shoulder_roll_joint"   : ACTUATOR_5020,
    "left_shoulder_yaw_joint"    : ACTUATOR_5020,
    "left_elbow_joint"           : ACTUATOR_5020,
    "left_wrist_roll_joint"      : ACTUATOR_5020,
    "left_wrist_pitch_joint"     : ACTUATOR_4010,
    "left_wrist_yaw_joint"       : ACTUATOR_4010,

    "right_shoulder_pitch_joint" : ACTUATOR_5020,
    "right_shoulder_roll_joint"  : ACTUATOR_5020,
    "right_shoulder_yaw_joint"   : ACTUATOR_5020,
    "right_elbow_joint"          : ACTUATOR_5020,
    "right_wrist_roll_joint"     : ACTUATOR_5020,
    "right_wrist_pitch_joint"    : ACTUATOR_4010,
    "right_wrist_yaw_joint"      : ACTUATOR_4010,
}

class XmlCustomActuator(XmlActuator):
  """Wrapper for XML-defined <motor> actuators."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.actuator_name = self._target_names[0]

    motor_type      = MATCHING_DICT[self.actuator_name]
    self.tau_acc    = torch.tensor(motor_type.tau_acc  ).unsqueeze(0)
    self.tau_br     = torch.tensor(motor_type.tau_br   ).unsqueeze(0)
    self.v1         = torch.tensor(motor_type.v1       ).unsqueeze(0)
    self.v2         = torch.tensor(motor_type.v2       ).unsqueeze(0)
    self.kp         = torch.tensor(motor_type.stiffness).unsqueeze(0)
    self.kd         = torch.tensor(motor_type.damping  ).unsqueeze(0)
    self.fs         = torch.tensor(motor_type.fs       ).unsqueeze(0)
    self.fd         = torch.tensor(motor_type.fd       ).unsqueeze(0)
    self.va         = torch.tensor(motor_type.va       ).unsqueeze(0)

    self.device_init = False


  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:

    if not self.device_init:
        self.tau_acc = self.tau_acc.to(cmd.pos.device)
        self.tau_br  = self.tau_br.to(cmd.pos.device)
        self.v1      = self.v1.to(cmd.pos.device)
        self.v2      = self.v2.to(cmd.pos.device)
        self.kp      = self.kp.to(cmd.pos.device)
        self.kd      = self.kd.to(cmd.pos.device)
        self.fs      = self.fs.to(cmd.pos.device)
        self.fd      = self.fd.to(cmd.pos.device)
        self.va      = self.va.to(cmd.pos.device)

        self.device_init = True

    q_des  = cmd.position_target
    # dq_des = torch.zeros_like(q_des)

    q_cur  = cmd.pos
    dq_cur = cmd.vel

    q_err = q_des - q_cur
    dq_err = -dq_cur #dq_des - dq_cur

    raw_torques = cmd.effort_target + self.kp * q_err + self.kd * dq_err
    return raw_torques

    # --- определяем режим: motoring / braking ---
    motoring_mask = (dq_cur * raw_torques) > 1e-6

    tau_max_0 = torch.where(motoring_mask, self.tau_acc, self.tau_br)

    # --- считаем |velocity| ---
    v_abs = torch.abs(dq_cur)

    # --- считаем лимит torque как функцию скорости ---
    # 3 зоны:
    # 1) v < v_x1 -> tau_max_0
    # 2) v_x1 <= v <= v_x2 -> linear drop
    # 3) v > v_x2 -> 0

    # линейная часть
    # избегаем деления на 0
    denom = torch.clamp(self.v2 - self.v1, min=1e-6)

    linear_scale = 1.0 - (v_abs - self.v1) / denom
    linear_scale = torch.clamp(linear_scale, 0.0, 1.0)

    tau_limit = torch.where(
        v_abs < self.v1,
        tau_max_0,
        tau_max_0 * linear_scale,
    )

    # при v > v_x2 -> 0
    tau_limit = torch.where(v_abs > self.v2, torch.zeros_like(tau_limit), tau_limit)

    # --- финальный клип ---
    clipped_torques = torch.clamp(raw_torques, -tau_limit, tau_limit)

    friction_torques = self.fs * torch.tanh(dq_cur / self.va) + self.fd * dq_cur
    return clipped_torques - friction_torques
  
@dataclass(kw_only=True)
class XmlCustomActuatorCfg(XmlActuatorCfg):
  """Wrap existing XML-defined <motor> actuators."""

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> XmlCustomActuator:
    return XmlCustomActuator(self, entity, target_ids, target_names)

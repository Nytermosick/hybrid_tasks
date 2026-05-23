from __future__ import annotations
import torch
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

from mjlab.managers.scene_entity_config import SceneEntityCfg
from hybrid_tasks.assets.robots import BODY_HEIGHT_DESIRED,\
                                       GAIT_PERIOD, GAIT_OFFSET,\
                                       GAIT_THRESHOLD, STANCE_PERIOD,\
                                       F_MAX_Z, MU,\
                                       G1_MASS, G1_BASE_INERTIA,\
                                       BASE_POS_KP, BASE_POS_KD, BASE_ORIENT_KP, BASE_ORIENT_KD,\
                                       COMMAND_STANDING_THRESHOLD

np.set_printoptions(precision=3, suppress=True, linewidth=2000)
torch.set_printoptions(precision=3, linewidth=2000)
e = torch.autograd.Variable(torch.Tensor())    # no equality

from qpth.qp import QPFunction, QPSolvers
import time
import mjlab.utils.lab_api.math as math_utils

class QPCfg():    
    def __init__(self, num_envs=1, device="cuda"):
        print("Init QPCFG")
        print("num_envs:", num_envs)

        self.qf = QPFunction(verbose=-1, check_Q_spd=False, solver=QPSolvers.PDIPM_BATCHED)
        
        self.fz_max = torch.tensor(F_MAX_Z, device=device)
        
        I3 = np.eye(3)
        O3 = np.zeros([3, 3])
        
        # mp = 1.0
        # mm = 1.0
        mp = 1.25
        mm = 1.11

        # Foot contact verticies
        fcv0 = np.array([+0.12 * mp, +0.030, -0.03])
        fcv1 = np.array([+0.12 * mp, -0.030, -0.03])
        fcv2 = np.array([-0.05 * mm, +0.025, -0.03])
        fcv3 = np.array([-0.05 * mm, -0.025, -0.03])

        Af = np.vstack([
            np.hstack([-1, 0, -MU]),
            np.hstack([+1, 0, -MU]),
            np.hstack([0, -1, -MU]),
            np.hstack([0, +1, -MU]),
        ])
        Aff = np.kron(np.eye(4), Af)
        
        gravity = [0, 0, 9.81, 0, 0, 0]
        p_des = [0, 0, BODY_HEIGHT_DESIRED]  # same as in rewards
        
        # Torch Arrays
        self.I3 = torch.stack([torch.eye(3, device=device)] * num_envs, dim=0).float()
        self.O18x6 = torch.zeros([num_envs, 18, 6], device=device)

        self.Ib = torch.stack([torch.tensor(G1_BASE_INERTIA, device=device)] * num_envs, dim=0)

        Au_top = np.hstack([1.0 / G1_MASS * I3, O3, 1.0 / G1_MASS * I3, O3])
        self.Au_top = torch.stack([torch.tensor(Au_top, device=device)] * num_envs, dim=0).float()

        self.p_des = torch.stack([torch.tensor(p_des, device=device)] * num_envs, dim=0)
        self.mass = torch.stack([torch.tensor(G1_MASS, device=device)] * num_envs, dim=0)

        # Tensors converted from numpy has double datatype (Float64). Pytorch default is float (Float32)
        # Concatenates sequence of tensors along a new dimension.
        self.Kpl = torch.stack([torch.tensor(BASE_POS_KP, device=device)] * num_envs, dim=0).float()
        self.Kdl = torch.stack([torch.tensor(BASE_POS_KD, device=device)] * num_envs, dim=0).float()
        self.Kpa = torch.stack([torch.tensor(BASE_ORIENT_KP, device=device)] * num_envs, dim=0).float()
        self.Kda = torch.stack([torch.tensor(BASE_ORIENT_KD, device=device)] * num_envs, dim=0).float()

        self.fcv0 = torch.stack([torch.tensor(fcv0, device=device)] * num_envs, dim=0).float()
        self.fcv1 = torch.stack([torch.tensor(fcv1, device=device)] * num_envs, dim=0).float()
        self.fcv2 = torch.stack([torch.tensor(fcv2, device=device)] * num_envs, dim=0).float()
        self.fcv3 = torch.stack([torch.tensor(fcv3, device=device)] * num_envs, dim=0).float()

        self.Aff = torch.stack([torch.tensor(Aff, device=device)] * num_envs, dim=0).float()
        self.gravity = torch.stack([torch.tensor(gravity, device=device)] * num_envs, dim=0).float()

        self.quat_yaw_init = torch.zeros([num_envs, 4], device=device)
        self.quat_yaw_init[:, 0] = 1.0    # w in quaternion
        

def solveQP(
    env: ManagerBasedRlEnv,
    qpcfg: QPCfg,
    a_policy: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # ts = time.perf_counter()
    #     
    robot: Entity = env.scene["robot"]
    command = env.command_manager.get_command("twist")

    c = gaitStanceWithCommand(
        env,
        command,
        period=GAIT_PERIOD,
        offset=GAIT_OFFSET,
        threshold=GAIT_THRESHOLD,
        command_threshold=COMMAND_STANDING_THRESHOLD,
    )
    
    qRwb = robot.data.root_link_quat_w
    qRwbz = math_utils.yaw_quat(qRwb)
    Rwb = math_utils.matrix_from_quat(qRwb)
    RwbT = torch.transpose(Rwb, 1, 2)
    
    I3 = qpcfg.I3
    O18x6 = qpcfg.O18x6
    
    p_des = qpcfg.p_des.clone()
    p_des[:, :2] += env.scene.env_origins[:, :2]
    
    v_des = torch.zeros_like(robot.data.body_link_lin_vel_w[:, 0])
    w_des = torch.zeros_like(robot.data.body_link_ang_vel_w[:, 0])
    
    v_des[:, 0:2] = command[:, 0:2]
    v_des = math_utils.quat_apply(qRwbz, v_des) # TODO: Или на qRwb?

    w_des[:, 2] = command[:, 2]


    qpcfg.quat_yaw_init = math_utils.quat_mul(
        math_utils.quat_from_euler_xyz(
            w_des[:, 0],
            w_des[:, 1],
            w_des[:, 2] * env.step_dt
        ),
        torch.where(
            env.episode_length_buf.unsqueeze(-1) == 0,
            math_utils.yaw_quat(qRwb),
            qpcfg.quat_yaw_init
        )
    )
    rpy_des = torch.zeros((env.num_envs, 3), device=env.device)
    quat_des = math_utils.quat_from_euler_xyz(rpy_des[:, 0], rpy_des[:, 1], rpy_des[:, 2])    # XYZ order!!!
    quat_des = math_utils.quat_mul(qpcfg.quat_yaw_init, quat_des)
    
    Kpl = qpcfg.Kpl
    Kdl = qpcfg.Kdl
    Kpa = qpcfg.Kpa
    Kda = qpcfg.Kda

    fcv0 = qpcfg.fcv0
    fcv1 = qpcfg.fcv1
    fcv2 = qpcfg.fcv2
    fcv3 = qpcfg.fcv3

    Aff = qpcfg.Aff
    mass = qpcfg.mass
    
    p_act = robot.data.body_link_pos_w[:, 0]
    v_act = robot.data.body_link_lin_vel_w[:, 0] # TODO: we cant use in real robot without estimator
    w_act = robot.data.body_link_ang_vel_w[:, 0]
    
    ori_err = math_utils.quat_box_minus(quat_des, qRwb)    # quat error = quat_des - quat_act
    # print("ori error:", ori_err)
    
    Ib = qpcfg.Ib
    Iw = torch.bmm(Rwb.float(), torch.bmm(Ib.float(), RwbT.float()))
    
    foot_names = ("left_foot", "right_foot")
    asset_cfg = SceneEntityCfg("robot", site_names=foot_names)
    asset_cfg.resolve(env.scene)
    
    foot_site_ids = asset_cfg.site_ids
    foot_pos_w = robot.data.site_pos_w[:, foot_site_ids, :]
    foot_quat_w = robot.data.site_quat_w[:, foot_site_ids, :]

    left_foot_pos  = foot_pos_w[:, 0]    # Left foot position in world frame
    right_foot_pos = foot_pos_w[:, 1]    # Right foot position in world frame
    
    rw_left = left_foot_pos - p_act
    rw_right = right_foot_pos - p_act

    left_foot_quat  = foot_quat_w[:, 0]   # Left foot orientation in world frame
    right_foot_quat = foot_quat_w[:, 1]   # Right foot orientation in world frame
    
    Au_top = qpcfg.Au_top
    Au_bot = torch.cat([
        torch.linalg.solve(Iw, math_utils.skew_symmetric_matrix(rw_left)), I3,\
        torch.linalg.solve(Iw, math_utils.skew_symmetric_matrix(rw_right)), I3],\
     dim=2)
    Au = torch.cat([Au_top, Au_bot], dim=1)
    # print("Au:\n", Au)

    # TODO: Really need?
    dl = torch.zeros_like(left_foot_pos)
    dl[:, 0] = 0.025
    p_des[:, :2] = ((left_foot_pos + right_foot_pos + 2 * dl) / 2.0)[:, :2]
    
    a_des_lin = Kpl * (p_des - p_act) + Kdl * (v_des - v_act)
    a_des_ang = Kpa * ori_err + Kda * (w_des - w_act)
    a_des = torch.cat([a_des_lin, a_des_ang], dim=1)
    # a_des[:, :3] += math_utils.quat_apply(qRwbz, a_policy[:, :3])  # a_policy should be in projected frame!
    # a_des[:, 3:] += math_utils.quat_apply(qRwbz, a_policy[:, 3:])  # a_policy should be in projected frame!

    # print("p_act:\n", p_act)
    # print("v_act:\n", v_act)
    # print("o_act:\n", robot.data.root_quat_w)
    # print("w_act:\n", w_act)
    
    # print("p_des:\n", p_des)
    # print("v_des:\n", v_des)
    # print("o_des:\n", quat_des)
    # print("rpy_des:\n", rpy_des)
    # print("w_des:\n", w_des)
    
    # print("a_des:\n", a_des)
    # print("a_policy:\n", a_policy)
    
    gravity = qpcfg.gravity
    a = (a_des + gravity).unsqueeze(2)
    
    # print("a:", a)
    # print("a shape:", a.shape)
    # print("Au shape:", Au.shape)
    # print("gravity shape:", gravity.shape)
    
    g = torch.bmm(torch.transpose(-a.float(), 1, 2), Au.float())
    g = torch.transpose(g, 1, 2)
    H = torch.bmm(torch.transpose(Au, 1, 2), Au) + 1e-7 * torch.eye(12, device=env.device).unsqueeze(0).expand(env.num_envs, -1, -1)
    
    A = torch.zeros([env.num_envs, 18 * 2, 12], device=env.device)
    ub = torch.zeros([env.num_envs, 18 * 2], device=env.device)

    B = torch.cat([
        torch.cat([I3, I3, I3, I3], dim=2),
        torch.cat([tskew(math_utils.quat_apply(left_foot_quat, fcv0)),
                   tskew(math_utils.quat_apply(left_foot_quat, fcv1)),
                   tskew(math_utils.quat_apply(left_foot_quat, fcv2)),
                   tskew(math_utils.quat_apply(left_foot_quat, fcv3))], dim=2),
    ], dim=1)
    Binv0 = torch.linalg.pinv(B)

    B = torch.cat([
        torch.cat([I3, I3, I3, I3], dim=2),
        torch.cat([tskew(math_utils.quat_apply(right_foot_quat, fcv0)),
                   tskew(math_utils.quat_apply(right_foot_quat, fcv1)),
                   tskew(math_utils.quat_apply(right_foot_quat, fcv2)),
                   tskew(math_utils.quat_apply(right_foot_quat, fcv3))], dim=2),
    ], dim=1)
    Binv1 = torch.linalg.pinv(B)

    D0 = torch.bmm(Aff.float(), Binv0.float())
    D1 = torch.bmm(Aff.float(), Binv1.float())

    Acwc0 = torch.cat([
        D0,
        torch.stack([torch.tensor([0, 0, +1, 0, 0, 0], device=env.device)] * env.num_envs, dim=0).unsqueeze(1),
        torch.stack([torch.tensor([0, 0, -1, 0, 0, 0], device=env.device)] * env.num_envs, dim=0).unsqueeze(1),
    ], dim=1)

    Acwc1 = torch.cat([
        D1,
        torch.stack([torch.tensor([0, 0, +1, 0, 0, 0], device=env.device)] * env.num_envs, dim=0).unsqueeze(1),
        torch.stack([torch.tensor([0, 0, -1, 0, 0, 0], device=env.device)] * env.num_envs, dim=0).unsqueeze(1),
    ], dim=1)
    
    A = torch.cat([
        torch.cat([Acwc0, O18x6], dim=2),
        torch.cat([O18x6, Acwc1], dim=2),
    ], dim=1)

    ub[:, 16] = qpcfg.fz_max * c[:, 0]
    ub[:, 34] = qpcfg.fz_max * c[:, 1]
    
    # print("g shape:", g.shape)
    # print("H shape:", H.shape)
    # print("A shape:", A.shape)
    # print("ub shape:", ub.shape)
    
    # print("g device:", g.device)
    # print("H device:", H.device)
    # print("A device:", A.device)
    # print("ub device:", ub.device)
    
    # print("g:\n", g)
    # print("H:\n", H)
    # print("A:\n", A)
    # print("ub:\n", ub)
    
    # Input matrices must be this shape: g,u: [num_batches, n], H: [num_batces, n, n], A: [num_batces, m, n]
    # OR! If num_batces = 1, then [1, n] may be squeezed to [n] with .squeeze(0). Same for matrices
    # ALL DATA MUST BE DOUBLE TYPE!!! (FLOAT64)
    tsolver = time.perf_counter()
    f = qpcfg.qf(H.double(), g.double().squeeze(2), A.double(), ub.double(), e, e).unsqueeze(2).float()
    tsolver = time.perf_counter() - tsolver
    
    # print("f_opt:\n", f.squeeze(2).cpu().numpy())
    
    left_wrench = f[:, 0 * 6:0 * 6 + 6, :]
    right_wrench = f[:, 1 * 6:1 * 6 + 6, :]
    
    return left_wrench, right_wrench


def gaitStance(
    env: ManagerBasedRlEnv,
    period: float,
    offset: list[float],
    threshold: float = 0.5,
) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase.view(env.num_envs, -1, 1))
    leg_phase = torch.cat(phases, dim=2)

    # print("global phase:", global_phase)
    # print("phases:", phases)
    # print("leg phase:", leg_phase)

    is_stance = (leg_phase < threshold).to(env.device).squeeze(1)  # [num_envs, 2]

    return is_stance

def gaitStanceWithCommand(
    env: ManagerBasedRlEnv,
    command: torch.Tensor,
    period: float,
    offset: list[float],
    threshold: float = 0.5,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    is_stance = gaitStance(env, period=period, offset=offset, threshold=threshold)
    total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    standing = total_command <= command_threshold
    return torch.where(standing.unsqueeze(1), torch.ones_like(is_stance), is_stance)


def gaitSwingPhase(
    env: ManagerBasedRlEnv,
    period: float,
    offset: list[float],
    threshold: float = 0.5,
) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    
    phases = []
    for offset_ in offset:
        phase = (global_phase+offset_) % 1.0
        phase = (phase >= threshold) * (phase-threshold) / (1.0-threshold)    # swing
        phases.append(phase.view(env.num_envs, -1))
    return torch.cat(phases, dim=1)


def gaitStancePhase(
    env: ManagerBasedRlEnv,
    period: float,
    offset: list[float],
    threshold: float = 0.5,
) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    
    phases = []
    for offset_ in offset:
        phase = (global_phase+offset_) % 1.0
        phase = (phase < threshold) * phase / threshold    # stance
        phases.append(phase.view(env.num_envs, -1))
    return torch.cat(phases, dim=1)


def gaitPhase(
    env: ManagerBasedRlEnv,
    period: float,
    offset: list[float],
) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    
    phases = []
    for offset_ in offset:
        phase = (global_phase+offset_) % 1.0
        phases.append(phase.view(env.num_envs, -1, 1))
    leg_phase = torch.cat(phases, dim=2)
    
    return leg_phase.to(env.device).squeeze(1)    # [num_envs, 4]

def tskew(v) -> torch.Tensor:
    return math_utils.skew_symmetric_matrix(v)

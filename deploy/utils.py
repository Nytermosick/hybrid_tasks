import re
import os
import numpy as np

class ControllerError(Exception):
    pass

def find_gamepad_event():
    """
    Автоматически ищет event-файл джойстика (геймпада).
    Возвращает путь вида: /dev/input/eventXX
    """

    devs_file = "/proc/bus/input/devices"

    if not os.path.exists(devs_file):
        raise ControllerError("Не найден /proc/bus/input/devices")

    with open(devs_file, "r") as f:
        content = f.read()

    # Разбиваем на блоки одно устройство = один блок
    blocks = content.strip().split("\n\n")

    for block in blocks:
        # Ищем имя устройства (строка N: Name="...")
        name_match = re.search(r'Name="([^"]+)"', block)
        if not name_match:
            continue
        name = name_match.group(1)

        # Определяем, похоже ли это на геймпад
        if not any(keyword in name.lower() for keyword in [
            "sony", "dualshock", "dualsense", "xbox", "controller", "gamepad", "joystick"
        ]):
            continue

        # Ищем строку с handlers
        handlers_match = re.search(r'H: Handlers=(.*)', block)
        if not handlers_match:
            continue
        handlers = handlers_match.group(1).split()

        # Нам нужен eventXX
        for h in handlers:
            if h.startswith("event"):
                return f"/dev/input/{h}"

    raise ControllerError("Event-файл контроллера не найден! Проверьте подключение")

def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Повернуть вектор v обратным кватернионом q.
    q: (w, x, y, z) формы (4,)
    v: (x, y, z) формы (3,)
    return: (3,)
    """
    q = np.asarray(q, dtype=float).reshape(4)
    v = np.asarray(v, dtype=float).reshape(3)

    w = q[0]
    q_vec = q[1:]               # (3,)

    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * (2.0 * w)
    c = q_vec * (2.0 * np.dot(q_vec, v))

    return a - b + c            # знак "минус b" → именно inverse-поворот

def compute_gait_phase(
    time: float | np.ndarray,
    period: float,
    command: np.ndarray | None = None,
    command_threshold: float = 0.1,
) -> np.ndarray:
    global_phase = np.asarray(time, dtype=float) % period / period
    phase = np.stack(
        (
            np.sin(global_phase * np.pi * 2.0),
            np.cos(global_phase * np.pi * 2.0),
        ),
        axis=-1,
    )

    if command is None:
        return phase

    command_norm = np.linalg.norm(np.asarray(command, dtype=float), axis=-1)
    stand_mask = command_norm < command_threshold
    return np.where(np.expand_dims(stand_mask, axis=-1), np.zeros_like(phase), phase)

def compute_gait_stance(
    time: float | np.ndarray,
    period: float,
    offset: list[float] | np.ndarray,
    threshold: float,
    command: np.ndarray | None = None,
    command_threshold: float = 0.1,
) -> np.ndarray:
    global_phase = np.asarray(time, dtype=float) % period / period
    leg_phase = (np.expand_dims(global_phase, axis=-1) + np.asarray(offset, dtype=float)) % 1.0
    is_stance = leg_phase < threshold

    if command is None:
        return is_stance

    command = np.asarray(command, dtype=float)
    stand_mask = np.linalg.norm(command[..., :2], axis=-1) + np.abs(command[..., 2]) <= command_threshold
    return np.where(np.expand_dims(stand_mask, axis=-1), np.ones_like(is_stance), is_stance)

def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def quat_error(q1, q2):
    """Computes the rotation difference between two quaternions.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        Angular error between input quaternions in radians.
    """
    quat_diff = quat_mul(q1, quat_conjugate(q2))
    return axis_angle_from_quat(quat_diff)

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Перемножение двух кватернионов (w, x, y, z) -> (w, x, y, z).
    Ожидаются одномерные массивы формы (4,).
    """
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)

    if q1.shape != (4,) or q2.shape != (4,):
        raise ValueError(f"Expected shape (4,), got {q1.shape} and {q2.shape}")

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([w, x, y, z], dtype=float)

def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Сопряжение кватерниона (w, x, y, z) -> (w, -x, -y, -z).
    Ожидается shape (4,).
    """
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

def axis_angle_from_quat(quat: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """
    quat: (w, x, y, z), shape (4,)
    return: axis-angle вектор, shape (3,)
    """
    q = np.asarray(quat, dtype=float).copy()
    if q.shape != (4,):
        raise ValueError(f"Expected shape (4,), got {q.shape}")

    # нормируем знак, чтобы w >= 0 (эквивалент q *= (1 - 2*(w<0)))
    if q[0] < 0.0:
        q = -q

    mag = np.linalg.norm(q[1:])
    half_angle = np.arctan2(mag, q[0])
    angle = 2.0 * half_angle

    if abs(angle) > eps:
        denom = np.sin(half_angle) / angle
    else:
        # Тейлорово приближение sin(θ/2)/θ ≈ 1/2 - θ^2/48 при θ→0
        denom = 0.5 - (angle * angle) / 48.0

    return q[1:4] / denom

def integrate_quat(q_des_curr, ang_vel_w, dt, eps=1e-12):
    """
    q_wxyz: текущий desired quaternion, [w,x,y,z], ориентация body в world
    omega_world: заданная угловая скорость в world frame, [wx,wy,wz], rad/s
    dt: шаг, s
    """
    w = np.asarray(ang_vel_w, dtype=float)
    wnorm = float(np.linalg.norm(w))
    if wnorm < eps:
        return q_des_curr  # почти ноль

    theta = wnorm * dt
    axis = w / wnorm

    dq = np.array([
        np.cos(theta * 0.5),
        axis[0] * np.sin(theta * 0.5),
        axis[1] * np.sin(theta * 0.5),
        axis[2] * np.sin(theta * 0.5),
    ], dtype=float)

    # omega в WORLD => dq слева
    q_next = quat_mul(dq, q_des_curr)
    return q_next / np.linalg.norm(q_next)

def quat_apply(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    Apply a quaternion rotation to a vector (NumPy).
    quat: (..., 4) in (w, x, y, z)
    vec : (..., 3) in (x, y, z)
    return: (..., 3)
    """
    # запоминаем целевую форму вывода
    out_shape = vec.shape

    q = np.asarray(quat, dtype=float).reshape(-1, 4)   # (N,4)
    v = np.asarray(vec,  dtype=float).reshape(-1, 3)   # (N,3)

    xyz = q[:, 1:]                                     # (N,3)
    t = 2.0 * np.cross(xyz, v)                         # (N,3)
    out = v + q[:, 0:1] * t + np.cross(xyz, t)         # (N,3)

    return out.reshape(out_shape)

def quat_to_R_wxyz(q):
    """q: iterable [w, x, y, z] -> 3x3 rotation matrix."""
    q = np.asarray(q, dtype=float)
    # нормализуем, чтобы избежать накопления ошибки
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    # x, y = 0, 0

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)]
    ], dtype=float)
    return R

def yaw_quat_from_quat(q):
    w, x, y, z = q  # wxyz
    
    # yaw extraction
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    
    # yaw-only quaternion
    return np.array([
        np.cos(yaw/2),
        0.0,
        0.0,
        np.sin(yaw/2)
    ])

def sqrt_positive_part(x: np.ndarray) -> np.ndarray:
    # max(x, 0) и корень — чтобы не уехать в комплексные из-за численного шума
    return np.sqrt(np.maximum(x, 0.0))

def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """
    R: (3,3) — вращательная матрица
    return: (4,) — кватернион (w, x, y, z)
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"Invalid rotation matrix shape {R.shape}.")

    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    q_abs = sqrt_positive_part(np.array([
        1.0 + m00 + m11 + m22,   # w-кандидат
        1.0 + m00 - m11 - m22,   # x-кандидат
        1.0 - m00 + m11 - m22,   # y-кандидат
        1.0 - m00 - m11 + m22,   # z-кандидат
    ], dtype=float))

    # кандидаты (каждый — кватернион, домноженный на соответствующий компонент)
    quat_by_rijk = np.array([
        [ q_abs[0]**2,   m21 - m12,   m02 - m20,   m10 - m01],  # w
        [ m21 - m12,     q_abs[1]**2, m10 + m01,   m02 + m20],  # x
        [ m02 - m20,     m10 + m01,   q_abs[2]**2, m12 + m21],  # y
        [ m10 - m01,     m20 + m02,   m21 + m12,   q_abs[3]**2] # z
    ], dtype=float)

    # нормируем кандидатов делением на 2*max(q_abs, 0.1) — как в исходнике
    denom = 2.0 * max(q_abs.max(), 0.1)
    quat_candidates = quat_by_rijk / denom  # shape (4,4)

    # выбираем «лучшего» по наибольшему q_abs (устойчивее численно)
    best = int(np.argmax(q_abs))
    q = quat_candidates[best]  # (4,) как (w,x,y,z)

    # опционально можно нормировать (полезно при числ. шуме)
    n = np.linalg.norm(q)
    return q / n if n > 0 else q

def so3_error(R, Rd):
    # Rd * R^T
    Re = Rd @ R.T
    return 0.5 * np.array([
        Re[2,1] - Re[1,2],
        Re[0,2] - Re[2,0],
        Re[1,0] - Re[0,1]
    ])

def euler_from_quat(q):
    w, x, y, z = q  # wxyz

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])

def wrap_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def yaw_from_quat(quat):
    return euler_from_quat(quat)[2]


def yaw_quat_from_yaw(yaw):
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=float)
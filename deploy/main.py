import os
import time
from pynput import keyboard
import argparse
import threading
import numpy as np
from evdev import InputDevice, ecodes

from utils import find_gamepad_event
from robot_env import G1_Env
from hybrid_tasks.assets.robots import CONTROL_DT, G1_NUM_MOTOR
from simple_controller import SimpleController
from QP_controller import QPController

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

### НАСТРОЙКА ПАРСЕРА ###
parser = argparse.ArgumentParser(
    allow_abbrev=False
)

parser.add_argument(
    "--interface",
    type=str,
    default="lo",
    help="Если указывается, то ChannelFactoryInitialize инициализируется указанным интерфейсом. Если не указывается, то по-умолчанию берётся 'lo' (запуск в симуляции)"
)

parser.add_argument(
    "--joystick_on",
    action="store_true",
    help="Для подключения управления с помощью джойстика пропишите данный аргумент"
)

parser.add_argument(
    "--policy_path",
    type=str,
    # default="logs/rsl_rl/g1_vanilla_walk/vanilla/policy.onnx", # without yaw_error
    # default="logs/rsl_rl/g1_qp_without_acc_walk/2026-05-27_21-22-37_finetuned_for_82k/policy.onnx", # without yaw_error
    # default="logs/rsl_rl/g1_qp_without_acc_walk/2026-05-29_16-01-00_qp_with_yaw_error_finetuned_39k/policy.onnx", # with yaw_error
    # default="logs/rsl_rl/g1_vanilla_walk/2026-05-30_17-57-52_30k/policy.onnx", # without yaw_error
    default="logs/rsl_rl/g1_qp_without_acc_walk/2026-05-31_01-32-09_30k/policy.onnx", # without yaw_error
    help="Путь до файла политики."
)

args = parser.parse_args()

def gamepad_listener(): # HARDCODED FOR DUALSHOCK4
    global mode, gamepad

    for event in gamepad.read_loop():
        # кнопки
        if event.type == ecodes.EV_KEY and event.value == 1:  # key down
            if event.code == 304:       # X
                if mode == 3 or mode == 2:
                    continue
                mode = 1
            elif event.code == 308:     # Square
                if mode == 3:
                    continue
                mode = 2
            elif event.code == 307:     # Triangle
                mode = 3

def keyboard_listener(key):
    global mode
    try:
        k = key.char.lower()
    except:
        return

    if k == '1':
        if mode == 3 or mode == 2:
            return
        mode = 1
    elif k == '2':
        if mode == 3:
            return
        mode = 2
    elif k == '3':
        mode = 3
    else:
        return

### НАСТРОЙКА ДЖОЙСТИКА ###
GAMEPAD_AXIS_COMMANDS = (
    {"cmd_idx": 0, "axis": 1, "sign": -1.0, "max_abs": 1.1},  # left stick Y -> vx
    {"cmd_idx": 1, "axis": 0, "sign": -1.0, "max_abs": 0.3},  # left stick X -> vy
    {"cmd_idx": 2, "axis": 3, "sign": -1.0, "max_abs": 1.5},  # right stick X -> wz
)
GAMEPAD_DEADZONE = 0.08
COMMAND_RATE_LIMIT = np.array([0.7, 1.2, 3.0])
target_velocity_commands = np.zeros(3, dtype=float)

def normalize_axis(value, min_raw, max_raw, *, max_abs, sign=1.0, deadzone=GAMEPAD_DEADZONE):
    value = max(min(value, max_raw), min_raw)
    normalized = 2.0 * (value - min_raw) / (max_raw - min_raw) - 1.0
    normalized *= sign
    if abs(normalized) < deadzone:
        return 0.0
    normalized = (abs(normalized) - deadzone) / (1.0 - deadzone) * (1.0 if normalized > 0.0 else -1.0)
    return normalized * max_abs

def update_velocity_commands_from_gamepad():
    global target_velocity_commands
    for axis_cfg in GAMEPAD_AXIS_COMMANDS:
        abs_info = gamepad.absinfo(axis_cfg["axis"])
        target_velocity_commands[axis_cfg["cmd_idx"]] = normalize_axis(
            abs_info.value,
            abs_info.min,
            abs_info.max,
            max_abs=axis_cfg["max_abs"],
            sign=axis_cfg["sign"],
        )
    # current = np.asarray(robot_env.velocity_commands, dtype=float)
    # max_delta = np.asarray(COMMAND_RATE_LIMIT, dtype=float) * CONTROL_DT
    # next_command = current + np.clip(target_velocity_commands - current, -max_delta, max_delta)
    # robot_env.velocity_commands[:] = next_command.tolist()
    robot_env.velocity_commands[:] = target_velocity_commands.tolist()

if args.joystick_on:
    device_path = find_gamepad_event()
    gamepad = InputDevice(device_path)
    gamepad_listener_thread = threading.Thread(target=gamepad_listener, daemon=True)
    gamepad_listener_thread.start()

keyboard_listener_thread = keyboard.Listener(on_press=keyboard_listener)
keyboard_listener_thread.start()

mode = 0
velocity_commands = [0.0, 0.0, 0.0]

interface = args.interface
robot_scene = os.path.join(ROOT_DIR, "external", "unitree_mujoco", "unitree_robots", "g1", "g1_29dof.xml")
robot_env = G1_Env(interface, robot_scene, control_dt=CONTROL_DT, velocity_commands=velocity_commands,
                   obs_dim=98, history_len=1, action_dim=G1_NUM_MOTOR)

controller = QPController(policy_path=args.policy_path, dt=CONTROL_DT)
# controller = SimpleController(policy_path=args.policy_path, dt=CONTROL_DT)

band_enabled = True if interface == "lo" else False # Only for Simulator!

print("DDS fabric and controller are initialized!\n")

robot_env.check_angles()

print("WARNING: Please ensure there are no stacles around the robot.")

print("""
Press 1 on keyboard or X on gamepad to move robot to initial position
Then press 2 on keyboard or SQUARE on gamepad to enable the policy
Press 3 on keyboard or TRIANGLE on gamepad for enabling damping mode (emergency disabling motors)
    """)

try:
    while True:
        if mode == 0:
            continue
        elif mode == 1:
            if not robot_env.in_default_pos:
                robot_env.move_to_default_pos()
            else:
                robot_env.stay_in_default_pos()
        elif mode == 2:
            step_start = time.perf_counter()

            if band_enabled:
                robot_env.disable_elastic_band()
                band_enabled = False

            if args.joystick_on:
                update_velocity_commands_from_gamepad()
            print(f"Desired Velocity: X={robot_env.velocity_commands[0]:+.03f} Y={robot_env.velocity_commands[1]:+.03f} Z={robot_env.velocity_commands[2]:+.03f}")

            obs_data = robot_env.get_measurements()
            robot_env.check_angles(obs_data)
        
            tau_des, q_des, dq_des = controller.step(obs_data)

            robot_env.send_low_cmd_msg(tau=tau_des,
                                        q=q_des,
                                        dq=dq_des)
            
            time_until_next_step = CONTROL_DT - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        elif mode == 3:
            if band_enabled:
                robot_env.disable_elastic_band()
                band_enabled = False
            robot_env.enable_damping_mode()
            break
except:
    if band_enabled:
        robot_env.disable_elastic_band()
        band_enabled = False
    robot_env.enable_damping_mode()

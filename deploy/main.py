import os
import time
from pynput import keyboard
import argparse
import threading
from evdev import InputDevice, ecodes

from utils import find_gamepad_event
from robot_env import G1_Env
from hybrid_tasks.assets.robots import CONTROL_DT, G1_NUM_MOTOR
from simple_controller import SimpleController
# from QP_controller import QPController

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
    "--air_fix",
    action="store_true",
    help="Для тестов робота в воздухе"
)

parser.add_argument(
    "--joystick_on",
    action="store_true",
    help="Для подключения управления с помощью джойстика пропишите данный аргумент"
)

parser.add_argument(
    "--policy_path",
    type=str,
    default="logs/rsl_rl/g1_vanilla_walk/2026-05-19_19-00-21/policy.onnx",
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
def normalize(value, min_raw=0, max_raw=255, out_min=-0.5, out_max=0.5):
    """Нормализация диапазона raw → [-0.5, 0.5]"""
    value = max(min(value, max_raw), min_raw)
    norm = (value - min_raw) / (max_raw - min_raw)
    mapped = out_min + norm * (out_max - out_min)
    return mapped

if args.joystick_on:
    device_path = find_gamepad_event()
    gamepad = InputDevice(device_path)
    gamepad_listener_thread = threading.Thread(target=gamepad_listener, daemon=True)
    gamepad_listener_thread.start()

keyboard_listener_thread = keyboard.Listener(on_press=keyboard_listener)
keyboard_listener_thread.start()

mode = 0
velocity_commands = [0.0, 0.0, 1.0]

interface = args.interface
robot_scene = os.path.join(ROOT_DIR, "external", "unitree_mujoco", "unitree_robots", "g1", "g1_29dof.xml")
robot_env = G1_Env(interface, robot_scene, control_dt=CONTROL_DT, velocity_commands=velocity_commands,
                   obs_dim=98, history_len=1, action_dim=G1_NUM_MOTOR)
# controller = QPController(policy_path=args.policy_path, dt=CONTROL_DT, enable_swing_controller=True, air_fix=args.air_fix)
# controller = QPController(policy_path=args.policy_path, dt=CONTROL_DT, enable_swing_controller=False, air_fix=args.air_fix)
controller = SimpleController(policy_path=args.policy_path)

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
                robot_env.velocity_commands[0] = -normalize(gamepad.absinfo(1)[0])
                robot_env.velocity_commands[1] = -normalize(gamepad.absinfo(0)[0])
                robot_env.velocity_commands[2] = -normalize(gamepad.absinfo(3)[0])
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
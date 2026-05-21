import os
import sys
import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading
import argparse

# Ensure the local mujoco directory is first on sys.path so imports of
# config resolve to deploy/mujoco/config.py when the script is run directly.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
import config

parser = argparse.ArgumentParser(
    allow_abbrev=False
)

parser.add_argument(
    "--air_fix",
    action="store_true",
    help="Для тестов робота в воздухе"
)

args = parser.parse_args()

latest_msg = None

def lowstate_callback(msg: LowCmd_):
    global latest_msg
    latest_msg = msg

locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

if args.air_fix:
    config.ENABLE_ELASTIC_BAND = False

if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    band_attached_link = mj_model.body("torso_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

viewer.cam.distance = 2.5
viewer.cam.elevation = -20
viewer.cam.azimuth = 45

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu

time.sleep(0.2)

def SimulationThread():
    global mj_data, mj_model

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
    sub.Init(lowstate_callback)

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3], mj_data.xquat[band_attached_link], mj_data.cvel[band_attached_link, 3:6]
                )

        if not args.air_fix and latest_msg and latest_msg.motor_cmd[0].dq != 0.0:
            elastic_band.enable = False

        if latest_msg:
            for i in range(29):
                mj_data.ctrl[i] = (
                    latest_msg.motor_cmd[i].tau +
                    latest_msg.motor_cmd[i].kp * (latest_msg.motor_cmd[i].q - mj_data.sensordata[i]) +
                    latest_msg.motor_cmd[i].kd * (latest_msg.motor_cmd[i].dq- mj_data.sensordata[i + 29])
                )

        if args.air_fix:
            mj_data.qpos[0:3] = [0, 0, 0.793]
            mj_data.qpos[3:7] = [1, 0, 0, 0]
            mj_data.qvel[:6] = [0, 0, 0, 0, 0, 0]

        mujoco.mj_step(mj_model, mj_data)

        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.cam.lookat[:] = mj_data.xpos[mj_model.body("torso_link").id]
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":    
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()
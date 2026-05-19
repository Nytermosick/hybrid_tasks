from mjlab.tasks.registry import register_mjlab_task
from hybrid_tasks.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)
from .rl_cfg import unitree_g1_ppo_runner_cfg

from .rl_cfgs_custom import g1_vanilla_ppo_runner_cfg
from .env_cfgs_custom import g1_vanilla_walk_flat_env_cfg

# Original tasks

register_mjlab_task(
  task_id="Unitree-G1-Rough",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Custom tasks

register_mjlab_task(
  task_id="G1-Vanilla-Walk-Flat",
  env_cfg=g1_vanilla_walk_flat_env_cfg(),
  play_env_cfg=g1_vanilla_walk_flat_env_cfg(play=True),
  rl_cfg=g1_vanilla_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import *
from .rl_cfg import unitree_g1_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-Tracking",
  env_cfg=unitree_g1_flat_tracking_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-No-State-Estimation",
  env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-Standing",
  env_cfg=unitree_g1_flat_tracking_standing_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_standing_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-1307-Stage-I",
  env_cfg=unitree_g1_flat_tracking_standing_env_cfg_1307_stage_I(),
  play_env_cfg=unitree_g1_flat_tracking_standing_env_cfg_1307_stage_I(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-1307-Stage-II",
  env_cfg=unitree_g1_flat_tracking_standing_env_cfg_1307_stage_II(),
  play_env_cfg=unitree_g1_flat_tracking_standing_env_cfg_1307_stage_II(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-1307-Stage-III",
  env_cfg=unitree_g1_flat_tracking_standing_env_cfg_1307_stage_III(),
  play_env_cfg=unitree_g1_flat_tracking_standing_env_cfg_1307_stage_III(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-1307-Checkpoint",
  env_cfg=unitree_g1_flat_tracking_standing_env_cfg_1307_stage_I(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_tracking_standing_env_cfg_1307_stage_I(has_state_estimation=False,play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
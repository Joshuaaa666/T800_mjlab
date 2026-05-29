"""Motion mimic task configuration.

This module defines the base configuration for motion mimic tasks.
Robot-specific configurations are located in the config/ directory.

This is a re-implementation of BeyondMimic (https://beyondmimic.github.io/).

Based on https://github.com/HybridRobotics/whole_body_tracking
Commit: f8e20c880d9c8ec7172a13d3a88a65e3a5a88448
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from src.assets.robots.t800.t800 import T800_EE_BODY_NAMES
from src.tasks.tracking.tracking_env_cfg import _apply_t800_robot_cfg

import src.tasks.tracking.mdp as mdp
MotionStandingCommandCfg = mdp.MotionStandingCommandCfg
TolerantTermination = mdp.TolerantTermination

VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}

VELOCITY_RANGE_ADD = {
  "x": (-0.75, 0.75),
  "y": (-0.75, 0.75),
  "z": (-0.3, 0.3),
  "roll": (-0.78, 0.78),
  "pitch": (-0.78, 0.78),
  "yaw": (-1.17, 1.17),
}


def _apply_t800_standing_command_cfg(cfg: ManagerBasedRlEnvCfg) -> None:
  """T800-specific settings for MotionStandingCommand."""
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionStandingCommandCfg)
  # GRSI: 预采集的 T800 倒地姿态数据集 (4096种, 由 robot_data.csv 转换)
  motion_cmd.init_pos_file = "data/t800_init_states_8192.pth"
  # tracking_standing_weight 默认 (1.0, 1.0), 50%起身/50%跟踪交替训练
  motion_cmd.root_body_name = ("LINK_BASE",)
  motion_cmd.shoulders_body_names = (
    "LINK_SHOULDER_ROLL_L",
    "LINK_SHOULDER_ROLL_R",
  )
  motion_cmd.feet_body_names = (
    "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_R",
  )

  for name, _func, params in cfg.terminations["tracking_failure"].func.terms:
    if name == "ee_body_pos_z":
      params["body_names"] = T800_EE_BODY_NAMES
      break


def make_tracking_standing_env_cfg_base() -> ManagerBasedRlEnvCfg:
  """Create robot-agnostic standing tracking task configuration."""

  ##
  # Observations
  ##

  actor_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.25, n_max=0.25),
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"biased": True},
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
    ),
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b, params={"command_name": "motion"}
    ),
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b, params={"command_name": "motion"}
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "motion": MotionStandingCommandCfg(
      init_pos_file="",
      root_body_name=(),
      shoulders_body_names=(),
      feet_body_names=(),
      tracking_standing_weight=(1.0, 1.0),
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-0.1, 0.1),
      # Override in robot cfg.
      motion_file="",
      anchor_body_name="",
      body_names=(),
    )
  }

  ##
  # Events
  ##

  events: dict[str, EventTermCfg] = {
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": VELOCITY_RANGE},
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set in robot cfg.
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.01, 0.01),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards: dict[str, RewardTermCfg] = {
    "motion_global_root_pos": RewardTermCfg(
      func=mdp.motion_global_anchor_position_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_global_root_ori": RewardTermCfg(
      func=mdp.motion_global_anchor_orientation_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_pos": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_global_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 3.14},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-1),
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),

    "electrical_power_cost": RewardTermCfg(
      func=mdp.penalty_electrical_power_cost,
      weight=-10,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_KNEE_PITCH.*",)),
      },
    ),
    "penalty_relative_shoulder_high": RewardTermCfg(
      func=mdp.penalty_relative_shoulder_high,
      weight=-2.0,
      params={"command_name": "motion"},
    ),
    "penalty_relative_root_orientation": RewardTermCfg(
      func=mdp.penalty_relative_root_orientation,
      weight=-0.5,
      params={"command_name": "motion"},
    ),
    "penalty_xy_rate_before_stand": RewardTermCfg(
      func=mdp.penalty_xy_rate_before_stand,
      weight=-1.0,
      params={"command_name": "motion","stand_threshold":0.1},
    ),
  }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "tracking_failure": TerminationTermCfg(
      func=TolerantTermination(
        bad_tracking_time_threshold_s=3.0,
        command_name="motion",
        terms=[
          ("anchor_pos_z", mdp.bad_anchor_pos_z_only, {
            "command_name": "motion",
            "threshold": 0.25,
          }),
          ("anchor_ori", mdp.bad_anchor_ori, {
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "motion",
            "threshold": 0.8,
          }),
          ("ee_body_pos_z", mdp.bad_motion_body_pos_z_only, {
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": (),
          }),
        ],
      ),
      params={},
    )
  }

  ##
  # Assemble and return
  ##

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=2.8,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
      max_extra_envs=0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=350,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
  return cfg


def _finalize_t800_standing_cfg(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  _apply_t800_robot_cfg(cfg)
  _apply_t800_standing_command_cfg(cfg)
  return cfg


def make_tracking_standing_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create T800 standing tracking task configuration."""
  return _finalize_t800_standing_cfg(make_tracking_standing_env_cfg_base())


def _apply_1307_stage_I(cfg: ManagerBasedRlEnvCfg) -> None:
  terminations = cfg.terminations["tracking_failure"]
  term_func = terminations.func
  # 放宽终止阈值，给机器人更多倒地挣扎时间
  for name, _func, params in term_func.terms:
    if name == "anchor_pos_z":
      params["threshold"] = 0.8    # 原0.5 → 0.8
    if name == "anchor_ori":
      params["threshold"] = 1.2    # 原0.8 → 1.2
    if name == "ee_body_pos_z":
      params["threshold"] = 0.8    # 原0.4 → 0.8
  # 延长容错时间
  term_func.bad_tracking_time_threshold_s = 8.0  # 原3.0秒 → 8.0秒

  # ===== 新增：随机初始化，让机器人从地上出生 =====
  cfg.events.update({
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.1, 0.1), "y": (-0.1, 0.1),
          "z": (0.3, 0.5),  # 0.8+0.3=1.1m ~ 0.8+0.5=1.3m高空坠落摔地
        },
        "velocity_range": {
          "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.3, 0.3),
          "roll": (-0.8, 0.8), "pitch": (-0.8, 0.8), "yaw": (-0.8, 0.8),
        },
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),   # 官方无随机关节,完全靠GRSI提供多样性
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    ),
  })


def make_tracking_standing_env_cfg_1307_stage_I_base() -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_standing_env_cfg_base()
  _apply_1307_stage_I(cfg)
  return cfg


def make_tracking_standing_env_cfg_1307_stage_I() -> ManagerBasedRlEnvCfg:
  return _finalize_t800_standing_cfg(make_tracking_standing_env_cfg_1307_stage_I_base())


def _apply_1307_stage_II(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.rewards.update({
    "reward_center_of_mass": RewardTermCfg(
      func=mdp.reward_center_of_mass,
      weight=1.0,
      params={"command_name": "motion","sigma_com": 0.1},
    ),
  })

  terminations = cfg.terminations["tracking_failure"].func
  delete_term_names = {"anchor_pos_z", "ee_body_pos_z"}
  terminations.terms = [t for t in terminations.terms if t[0] not in delete_term_names]
  for name, func, params in terminations.terms:
    if name == "anchor_ori":
        params["threshold"] = 0.6
  import math
  terminations.terms.extend([
    ("anchor_pos",mdp.bad_anchor_pos, {
      "command_name": "motion",
      "threshold": 1.0,
    }),
    ("hip_dof", mdp.bad_hip_dof, {
    "command_name": "motion",
    "threshold": math.pi /6,
    }),
  ]),


def make_tracking_standing_env_cfg_1307_stage_II_base() -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_standing_env_cfg_base()
  _apply_1307_stage_II(cfg)
  return cfg


def make_tracking_standing_env_cfg_1307_stage_II() -> ManagerBasedRlEnvCfg:
  return _finalize_t800_standing_cfg(make_tracking_standing_env_cfg_1307_stage_II_base())


def _apply_1307_stage_III(cfg: ManagerBasedRlEnvCfg) -> None:
  _apply_1307_stage_II(cfg)
  cfg.events.update({
    "terrain": EventTermCfg(
      func=mdp.randomize_terrain,
      mode="reset",
      params={},
    ),
    # 继承 Stage I 的倒地初始化 + Stage III 的更强速度扰动
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.1, 0.1), "y": (-0.1, 0.1),
          "z": (0.3, 0.5),  # 高空坠落摔地 (同 Stage I)
        },
        "velocity_range": {
          "x": (-0.75, 0.75), "y": (-0.75, 0.75), "z": (-0.3, 0.3),
          "roll": (-0.78, 0.78), "pitch": (-0.78, 0.78), "yaw": (-1.17, 1.17),
        },
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": VELOCITY_RANGE_ADD},
    ),
    # GRSI文件已提供倒地姿态关节角度, 这里仅加微小扰动保留GRSI数据
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        # GRSI文件提供关节角度, (0.0,0.0)不覆盖 (同官方)
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    )
  })


def make_tracking_standing_env_cfg_1307_stage_III_base() -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_standing_env_cfg_base()
  _apply_1307_stage_III(cfg)
  return cfg


def make_tracking_standing_env_cfg_1307_stage_III() -> ManagerBasedRlEnvCfg:
  return _finalize_t800_standing_cfg(make_tracking_standing_env_cfg_1307_stage_III_base())
"""T800 flat tracking environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg

from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from src.tasks.tracking.tracking_standing_env_cfg import (
  make_tracking_standing_env_cfg,
  make_tracking_standing_env_cfg_1307_stage_I,
  make_tracking_standing_env_cfg_1307_stage_II,
  make_tracking_standing_env_cfg_1307_stage_III,
)


def _apply_play_overrides(cfg: ManagerBasedRlEnvCfg, *, standing: bool) -> None:
  cfg.episode_length_s = int(1e9)
  cfg.observations["actor"].enable_corruption = False
  # cfg.events.pop("push_robot", None)

  motion_cmd = cfg.commands["motion"]
  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  motion_cmd.sampling_mode = "start"

  if standing:
    cfg.terminations.pop("tracking_failure", None)
    motion_cmd.tracking_standing_weight = (1.0, 0.0)


def t800_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """T800 flat terrain motion tracking."""
  cfg = make_tracking_env_cfg()

  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    _apply_play_overrides(cfg, standing=False)

  return cfg


def t800_flat_tracking_standing_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """T800 flat terrain motion tracking with standing command."""
  cfg = make_tracking_standing_env_cfg()

  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    _apply_play_overrides(cfg, standing=True)

  return cfg


def t800_flat_tracking_standing_env_cfg_1307_stage_I(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_standing_env_cfg_1307_stage_I()
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )
  if play:
    _apply_play_overrides(cfg, standing=True)
  return cfg


def t800_flat_tracking_standing_env_cfg_1307_stage_II(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_standing_env_cfg_1307_stage_II()
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )
  if play:
    _apply_play_overrides(cfg, standing=True)
  return cfg


def t800_flat_tracking_standing_env_cfg_1307_stage_III(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_standing_env_cfg_1307_stage_III()
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )
  if play:
    _apply_play_overrides(cfg, standing=True)
  return cfg

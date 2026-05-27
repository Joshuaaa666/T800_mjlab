from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import math

from mjlab.utils.lab_api.math import quat_apply_inverse

from .commands import MotionCommand
from .rewards import _get_body_indexes

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


class TolerantTermination:
  def __init__(self, bad_tracking_time_threshold_s: float, command_name: str, terms: list):
    self.bad_tracking_time_threshold_s = bad_tracking_time_threshold_s
    self.command_name=command_name
    self.terms = terms
    self.bad_tracking_episode_length = None
    self.last_triggered_terms = {}

  def __call__(self, env):
    self.env = env
    if self.bad_tracking_episode_length is None:
      self.bad_tracking_episode_length = torch.zeros(
        env.num_envs,
        device=env.device,
        dtype=torch.long,
      )

    current_combined_bad = torch.zeros(
      env.num_envs,
      device=env.device,
      dtype=torch.bool,
    )
    command = cast(MotionCommand, env.command_manager.get_term(self.command_name))

    for term_name, func, params in self.terms:
      raw_bad = func(env, **params)
      current_combined_bad |= raw_bad
      self.last_triggered_terms[term_name] = raw_bad

    bad_tracking_max_episode_length = math.ceil(self.bad_tracking_time_threshold_s / env.step_dt)

    self.bad_tracking_episode_length = torch.where(
      current_combined_bad,
      self.bad_tracking_episode_length + 1,
      torch.zeros_like(self.bad_tracking_episode_length),
    )

    bad_tracking_terminate = torch.where(
        command.is_standing_task,
        self.bad_tracking_episode_length >= bad_tracking_max_episode_length,
        current_combined_bad
    )
    self.final_trigger_mask = bad_tracking_terminate
    return bad_tracking_terminate

  def reset(self, env_ids=None):
    if self.bad_tracking_episode_length is None:
      return

    if env_ids is None:
        self.bad_tracking_episode_length.zero_()
        mask = self.final_trigger_mask
    else:
        self.bad_tracking_episode_length[env_ids] = 0
        mask = self.final_trigger_mask[env_ids]

    for name, value in self.last_triggered_terms.items():
      self.env.extras[f"Episode_Termination/TolerantTermination/{name}"] = torch.count_nonzero(
          (value if env_ids is None else value[env_ids]) & mask
      ).item()
    self.env.extras["log"].update({
        f"Episode_Termination/TolerantTermination/{name}":
        torch.count_nonzero(
            (value if env_ids is None else value[env_ids]) & mask
        ).item()
        for name, value in self.last_triggered_terms.items()
    })

def bad_anchor_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
  )


def bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    > threshold
  )


def bad_anchor_ori(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  motion_projected_gravity_b = quat_apply_inverse(
    command.anchor_quat_w, asset.data.gravity_vec_w
  )

  robot_projected_gravity_b = quat_apply_inverse(
    command.robot_anchor_quat_w, asset.data.gravity_vec_w
  )

  return (
    motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
  ).abs() > threshold


def bad_motion_body_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.norm(
    command.body_pos_relative_w[:, body_indexes]
    - command.robot_body_pos_w[:, body_indexes],
    dim=-1,
  )
  return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.abs(
    command.body_pos_relative_w[:, body_indexes, -1]
    - command.robot_body_pos_w[:, body_indexes, -1]
  )
  return torch.any(error > threshold, dim=-1)

def bad_hip_dof(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.abs(command.joint_pos[:,[0,1,2,6,7,8]] - command.robot_joint_pos[:,[0,1,2,6,7,8]])
  return torch.any(error > threshold, dim=-1)
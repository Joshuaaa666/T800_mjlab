from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.entity import Entity
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost

def penalty_relative_shoulder_high(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return torch.sum(torch.square(command.body_pos_relative_w[:, command.shoulders_indexes, 2] - command.robot_body_pos_w[:, command.shoulders_indexes, 2]), dim=-1)
  
def penalty_relative_root_orientation(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.body_quat_relative_w[:,command.root_index], command.robot_body_quat_w[:,command.root_index]) ** 2
  return error.squeeze(-1)

def penalty_xy_rate_before_stand(
  env: ManagerBasedRlEnv,
  command_name: str,
  stand_threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.norm(command.prev_anchor_pos[...,:2] - command.robot_anchor_pos_w[...,:2], dim=1)
  diff = torch.norm((command.body_pos_relative_w[:, command.shoulders_indexes, 2] - command.robot_body_pos_w[:, command.shoulders_indexes, 2]), dim=-1)
  return torch.where(diff > stand_threshold, error, torch.tensor(0.0, device=error.device, dtype=error.dtype))
  
def penalty_action_rate_before_stand(
  env: ManagerBasedRlEnv,
  command_name: str,
  stand_threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
  diff = torch.norm((command.body_pos_relative_w[:, command.shoulders_indexes, 2] - command.robot_body_pos_w[:, command.shoulders_indexes, 2]), dim=-1)
  return torch.where(diff > stand_threshold, error, torch.tensor(0.0, device=error.device, dtype=error.dtype))

def knee_ground_contact_cost_before_stand(
  env: ManagerBasedRlEnv,
  command_name: str,
  stand_threshold: float,
  sensor_name: str
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  diff = torch.norm((command.body_pos_relative_w[:, command.shoulders_indexes, 2] - command.robot_body_pos_w[:, command.shoulders_indexes, 2]), dim=-1)
  sensor: ContactSensor = env.scene[sensor_name]
  force = torch.norm(sensor.data.force, dim=-1)
  error = torch.sum(force, dim=1)
  return torch.where(diff > stand_threshold, error, torch.tensor(0.0, device=error.device, dtype=error.dtype))

def penalty_hip_roll_yaw_before_stand(
  env: ManagerBasedRlEnv,
  command_name: str,
  stand_threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  diff = torch.norm((command.body_pos_relative_w[:, command.shoulders_indexes, 2] - command.robot_body_pos_w[:, command.shoulders_indexes, 2]), dim=-1)

  joint_pos = command.robot_joint_pos[:, [0,1,6,7]]
  # limits (rad)
  lower = torch.tensor([-2.0944, -0.0873, -2.0944, -1.5708], device=joint_pos.device)
  upper = torch.tensor([0.0, 1.5708, 0.0, 0.0873], device=joint_pos.device)

  lower_violation = torch.relu(lower - joint_pos)
  upper_violation = torch.relu(joint_pos - upper)

  error = torch.sum(lower_violation + upper_violation, dim=1)
  return torch.where(diff > stand_threshold, error, torch.tensor(0.0, device=error.device, dtype=error.dtype))

def penalty_electrical_power_cost(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize electrical power consumption of actuators."""

  asset: Entity = env.scene[asset_cfg.name]

  joint_ids, _ = asset.find_joints(asset_cfg.joint_names)
  actuator_ids, _ = asset.find_actuators(asset_cfg.joint_names)

  tau = asset.data.actuator_force[:, actuator_ids]
  qd = asset.data.joint_vel[:, joint_ids]

  mech = - tau * qd - 150
  mech_pos = torch.clamp(mech, min=0.0)  # ignore regenerative power

  cost = torch.sum((mech_pos / 500)**2, dim=1)

  return cost

def reward_center_of_mass(
  env: ManagerBasedRlEnv,
  command_name: str,
  sigma_com: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  asset: Entity = env.scene[asset_cfg.name]

  com_xy = asset.data.root_com_pos_w[:, :2]

  z_l = command.robot_body_pos_w[:, command.feet_indexes[0], 2]
  z_r = command.robot_body_pos_w[:, command.feet_indexes[1], 2]

  single_support = torch.abs(z_l - z_r) > 0.05

  foot_l_xy = command.robot_body_pos_w[:, command.feet_indexes[0], :2]
  foot_r_xy = command.robot_body_pos_w[:, command.feet_indexes[1], :2]

  cond = z_l > z_r
  lower_foot_xy = torch.where(cond.unsqueeze(-1), foot_r_xy, foot_l_xy)

  error_sq = torch.sum((com_xy - lower_foot_xy) ** 2, dim=-1)

  reward = torch.exp(-error_sq / (sigma_com ** 2))

  return reward * single_support
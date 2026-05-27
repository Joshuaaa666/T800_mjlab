from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch
import os

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))

# MJCF body name -> motion .npz body name (when exporters use URDF naming).
_MOTION_BODY_NAME_ALIASES = {
  "LINK_WAIST_YAW": "LINK_TORSO_YAW",
}


def _motion_body_name(name: str) -> str:
  return _MOTION_BODY_NAME_ALIASES.get(name, name)


def _motion_joint_indexes_from_npz(
  motion_file: str, robot_joint_names: tuple[str, ...]
) -> torch.Tensor:
  data = np.load(motion_file)
  if "joint_names" not in data:
    raise ValueError(
      f"Motion file {motion_file} has no 'joint_names'; cannot align joints by name."
    )
  names = [str(n) for n in data["joint_names"]]
  indexes: list[int] = []
  for name in robot_joint_names:
    if name not in names:
      raise KeyError(
        f"Joint '{name}' not in motion file. Available: {names}"
      )
    indexes.append(names.index(name))
  return torch.tensor(indexes, dtype=torch.long)


def _motion_body_indexes_from_npz(
  motion_file: str, body_names: tuple[str, ...]
) -> torch.Tensor:
  data = np.load(motion_file)
  if "body_names" not in data:
    raise ValueError(
      f"Motion file {motion_file} has no 'body_names'; cannot align bodies by name."
    )
  names = [str(n) for n in data["body_names"]]
  indexes: list[int] = []
  for name in body_names:
    motion_name = _motion_body_name(name)
    if motion_name not in names:
      raise KeyError(
        f"Body '{name}' (motion alias '{motion_name}') not in motion file. "
        f"Available: {names}"
      )
    indexes.append(names.index(motion_name))
  return torch.tensor(indexes, dtype=torch.long)

def minmaxnorm(value):
    return (value - value.min()) / (value.max() - value.min())

def compute_sampling_prob_softmin(energy, temperature=1.0):
    """
    指数 softmin 概率：p_i = exp(-energy_i / temperature) / sum(exp(-energy_j / temperature))
    """
    logits = -energy / temperature
    prob = torch.softmax(logits, dim=0)
    return prob

class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    body_indexes: torch.Tensor,
    joint_indexes: torch.Tensor | None = None,
    device: str = "cpu",
    compute_kinetic_energy: bool = False,
  ) -> None:
    data = np.load(motion_file)
    joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
    joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
    if joint_indexes is not None:
      joint_pos = joint_pos[:, joint_indexes]
      joint_vel = joint_vel[:, joint_indexes]
    self.joint_pos = joint_pos
    self.joint_vel = joint_vel
    self._body_pos_w = torch.tensor(
      data["body_pos_w"], dtype=torch.float32, device=device
    )
    self._body_quat_w = torch.tensor(
      data["body_quat_w"], dtype=torch.float32, device=device
    )
    self._body_lin_vel_w = torch.tensor(
      data["body_lin_vel_w"], dtype=torch.float32, device=device
    )
    self._body_ang_vel_w = torch.tensor(
      data["body_ang_vel_w"], dtype=torch.float32, device=device
    )
    self._body_indexes = body_indexes
    self.body_pos_w = self._body_pos_w[:, self._body_indexes]
    self.body_quat_w = self._body_quat_w[:, self._body_indexes]
    self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_indexes]
    self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_indexes]
    self.time_step_total = self.joint_pos.shape[0]
    fps_arr = data["fps"] if "fps" in data else np.array([50.0])
    self.fps = float(np.asarray(fps_arr).reshape(-1)[0])
    if compute_kinetic_energy:
      kinetic_energy = self.joint_vel.pow(2).sum(-1) + self._body_lin_vel_w.norm(2,dim=-1).pow(2).sum(-1) + self._body_ang_vel_w.norm(2,dim=-1).pow(2).sum(-1)
      self.kinetic_energy_prob = minmaxnorm(compute_sampling_prob_softmin(kinetic_energy.clamp(kinetic_energy[0])))


class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )
    motion_body_indexes = _motion_body_indexes_from_npz(
      self.cfg.motion_file, self.cfg.body_names
    ).to(self.device)
    motion_joint_indexes = _motion_joint_indexes_from_npz(
      self.cfg.motion_file, self.robot.joint_names
    ).to(self.device)

    self.motion = MotionLoader(
      self.cfg.motion_file,
      motion_body_indexes,
      joint_indexes=motion_joint_indexes,
      device=self.device,
      compute_kinetic_energy=(self.cfg.sampling_mode == "lke"),
    )
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self.bin_count = int(self.motion.time_step_total // (1 / env.step_dt)) + 1
    self.bin_failed_count = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    # Ghost model created lazily on first visualization (robot-only, not full scene).
    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_free_joint_q_adr: np.ndarray | None = None
    self._ghost_joint_q_adr: np.ndarray | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
    control_hz = 1.0 / env.step_dt
    self._motion_steps_per_control = max(1, int(round(self.motion.fps / control_hz)))
    self._ghost_viz_data: mujoco.MjData | None = None
    self._ghost_vopt = mujoco.MjvOption()
    self._ghost_vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    self._ghost_pert = mujoco.MjvPerturb()
    try:
      self._ensure_ghost_model(self.robot)
    except Exception as exc:  # noqa: BLE001
      print(f"[WARN] Ghost model prebuild failed ({exc}); will retry on first render.")

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.time_steps]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      current_bin_index = torch.clamp(
        (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1),
        0,
        self.bin_count - 1,
      )
      fail_bins = current_bin_index[env_ids][episode_failed]
      self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

    # Sample.
    sampling_probabilities = (
      self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
    )
    sampling_probabilities = torch.nn.functional.pad(
      sampling_probabilities.unsqueeze(0).unsqueeze(0),
      (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
      mode="replicate",
    )
    sampling_probabilities = torch.nn.functional.conv1d(
      sampling_probabilities, self.kernel.view(1, 1, -1)
    ).view(-1)

    sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

    sampled_bins = torch.multinomial(
      sampling_probabilities, len(env_ids), replacement=True
    )
    self.time_steps[env_ids] = (
      (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
      / self.bin_count
      * (self.motion.time_step_total - 1)
    ).long()

    # Update metrics.
    H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
    H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
    pmax, imax = sampling_probabilities.max(dim=0)
    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

  def _uniform_sampling(self, env_ids: torch.Tensor):
    self.time_steps[env_ids] = torch.randint(
      0, self.motion.time_step_total, (len(env_ids),), device=self.device
    )
    self.metrics["sampling_entropy"][:] = 1.0  # Maximum entropy for uniform.
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
    self.metrics["sampling_top1_bin"][:] = 0.5  # No specific bin preference.

  def _lke_sampling(self, env_ids: torch.Tensor):
    """
      Low-kinetic-energy sampling
    """
    self.time_steps[env_ids] = torch.multinomial(self.motion.kinetic_energy_prob, env_ids.numel(), replacement=True)

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    elif self.cfg.sampling_mode == "lke":
      self._lke_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos[env_ids] += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += rand_samples[:, :3]
    root_ang_vel[env_ids] += rand_samples[:, 3:]

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
    )
    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )

    root_state = torch.cat(
      [
        root_pos[env_ids],
        root_ori[env_ids],
        root_lin_vel[env_ids],
        root_ang_vel[env_ids],
      ],
      dim=-1,
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    self.robot.clear_state(env_ids=env_ids)

  def _motion_frame_index(self, env_id: int) -> int:
    """Motion frame index for ghost visualization.

    Uses the global ``common_step_counter`` so playback advances once per env
    step and is never reset by episode termination (which happens every step at
    large ``num_envs`` during early training and froze the old per-episode
    counter / ``time_steps``-based logic).
    """
    del env_id
    total = max(self.motion.time_step_total, 1)
    frame = int(self._env.common_step_counter) * self._motion_steps_per_control
    return frame % total

  def _motion_state_at_frame(self, env_id: int, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Root pose and joint positions from the reference clip at ``frame``."""
    origin = self._env.scene.env_origins[env_id, :3].detach().cpu().numpy()
    root_pos = self.motion.body_pos_w[frame, 0].detach().cpu().numpy() + origin
    root_quat = self.motion.body_quat_w[frame, 0].detach().cpu().numpy()
    joint_pos = self.motion.joint_pos[frame].detach().cpu().numpy()
    return root_pos, root_quat, joint_pos

  def _ensure_ghost_model(self, entity: Entity) -> None:
    if self._ghost_model is not None:
      return
    # Entity spec is attached to the scene; compile a detached copy.
    self._ghost_model = entity.spec.copy().compile()
    self._ghost_model.geom_rgba[:] = self._ghost_color
    self._ghost_viz_data = mujoco.MjData(self._ghost_model)
    ghost_indexing = entity._compute_indexing(self._ghost_model, self.device)
    self._ghost_free_joint_q_adr = ghost_indexing.free_joint_q_adr.cpu().numpy()
    self._ghost_joint_q_adr = ghost_indexing.joint_q_adr.cpu().numpy()

  def _add_ghost_mesh(self, visualizer: DebugVisualizer, qpos: np.ndarray, label: str) -> None:
    """Add ghost geometry using a matching MjModel/MjData pair."""
    assert self._ghost_model is not None
    assert self._ghost_viz_data is not None
    self._ghost_viz_data.qpos[:] = qpos
    mujoco.mj_forward(self._ghost_model, self._ghost_viz_data)
    if isinstance(visualizer, MujocoNativeDebugVisualizer):
      mujoco.mjv_addGeoms(
        self._ghost_model,
        self._ghost_viz_data,
        self._ghost_vopt,
        self._ghost_pert,
        mujoco.mjtCatBit.mjCAT_DYNAMIC.value,
        visualizer.scn,
      )
    else:
      visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=label)

  def _update_command(self):
    self.time_steps += self._motion_steps_per_control
    env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

    if self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      entity: Entity = self._env.scene[self.cfg.entity_name]
      self._ensure_ghost_model(entity)
      assert self._ghost_model is not None
      assert self._ghost_free_joint_q_adr is not None
      assert self._ghost_joint_q_adr is not None

      for batch in env_indices:
        frame = self._motion_frame_index(batch)
        root_pos, root_quat, joint_pos = self._motion_state_at_frame(batch, frame)
        qpos = np.zeros(self._ghost_model.nq, dtype=np.float64)
        qpos[self._ghost_free_joint_q_adr[0:3]] = root_pos
        qpos[self._ghost_free_joint_q_adr[3:7]] = root_quat
        qpos[self._ghost_joint_q_adr] = joint_pos

        self._add_ghost_mesh(visualizer, qpos, label=f"ghost_{batch}")

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
        desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos,
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
        current_anchor_quat = self.robot_anchor_quat_w[batch]
        current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=current_anchor_pos,
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  motion_file: str
  anchor_body_name: str
  body_names: tuple[str, ...]
  entity_name: str
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  sampling_mode: Literal["adaptive", "uniform", "start", "lke"] = "adaptive"

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)

def select_most_diverse_quaternions(quats, num_select):
    """
    quats: (N,4) torch.Tensor, assumed normalized
    num_select: int, number of quaternions to select
    """
    N = quats.shape[0]
    selected_idx = [torch.randint(0, N, (1,)).item()] 
    if num_select == 1:
        return quats[selected_idx]

    for _ in range(num_select - 1):
        remaining_idx = list(set(range(N)) - set(selected_idx))
        remaining_quats = quats[remaining_idx]  # (M,4)
        selected_quats = quats[selected_idx]    # (len(selected_idx),4)
        
        sim = torch.abs(remaining_quats @ selected_quats.T)  # |dot product|
        dist = 1 - sim  
        min_dist, _ = dist.min(dim=1) 

        next_idx_in_remaining = min_dist.argmax().item()
        selected_idx.append(remaining_idx[next_idx_in_remaining])
    
    return selected_idx

def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]

class MotionStandingCommand(MotionCommand):
  cfg: MotionStandingCommandCfg
  _env: ManagerBasedRlEnv

  def _build_default_init_robot_data(self, num_samples: int = 2048) -> dict:
    """Build standing reset poses from default height and current motion pose."""
    root = torch.zeros(num_samples, 13, device=self.device)
    root[:, 2] = 0.8
    root[:, 3] = 1.0  # quaternion w (xyzw)
    root[:, :2] += torch.randn(num_samples, 2, device=self.device) * 0.05
    dof_pos = self.motion.joint_pos[0].unsqueeze(0).repeat(num_samples, 1)
    dof_pos = dof_pos + torch.randn_like(dof_pos) * 0.02
    return {"robot_root_states_xyzw": root.cpu(), "dof_pos": dof_pos.cpu()}

  def __init__(self, cfg: MotionStandingCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.pos_file = cfg.init_pos_file
    if cfg.init_pos_file and os.path.exists(cfg.init_pos_file):
      try:
        self.init_robot_data = torch.load(
          cfg.init_pos_file, map_location=self.device, weights_only=False
        )
      except TypeError:
        self.init_robot_data = torch.load(
          cfg.init_pos_file, map_location=self.device
        )
    else:
      if cfg.init_pos_file:
        print(
          f"[WARN] init_pos_file not found ({cfg.init_pos_file!r}); "
          "using auto-generated T800 standing init poses."
        )
      self.init_robot_data = self._build_default_init_robot_data()
    self.most_diverse_idxs = select_most_diverse_quaternions(self.init_robot_data["robot_root_states_xyzw"], 2048)
    self.is_standing_task = torch.multinomial(
        torch.tensor(cfg.tracking_standing_weight, device=self.device),
        num_samples=self.num_envs,
        replacement=True
    ).bool()
    self.prev_anchor_pos = torch.zeros_like(self.robot_anchor_pos_w)
    self.current_anchor_pos = torch.zeros_like(self.robot_anchor_pos_w)
    self.root_index = _get_body_indexes(self, cfg.root_body_name)
    self.shoulders_indexes = _get_body_indexes(self, cfg.shoulders_body_names)
    self.feet_indexes = _get_body_indexes(self, cfg.feet_body_names)

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    elif self.cfg.sampling_mode == "lke":
      self._lke_sampling(env_ids)      
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    reset_standing_indices = torch.multinomial(
        torch.tensor(self.cfg.tracking_standing_weight, device=self.device),
        num_samples=env_ids.numel(),
        replacement=True
    ).bool()
    self.is_standing_task[env_ids] = reset_standing_indices

    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos[env_ids] += rand_samples[:, 0:3]
    sampled_init_ids = torch.tensor(self.most_diverse_idxs)[torch.randint(low=0, high=len(self.most_diverse_idxs), size=(env_ids.numel(),)).long()]
    sampled_init_root_states_xyzw = self.init_robot_data["robot_root_states_xyzw"][sampled_init_ids].to(self.device)
    sampled_init_dof_pos = self.init_robot_data["dof_pos"][sampled_init_ids].to(self.device)
    sampled_init_dof_vel = torch.zeros_like(sampled_init_dof_pos)
    
    root_pos[env_ids] = torch.where(
        reset_standing_indices.unsqueeze(1),
        torch.cat([
            root_pos[env_ids, :2],
            sampled_init_root_states_xyzw[:, 2:3]
        ], dim=1),
        root_pos[env_ids]
    )
    
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
    root_ori[env_ids] = torch.where(
        reset_standing_indices.unsqueeze(1),
        sampled_init_root_states_xyzw[:, 3:7],
        root_ori[env_ids]
    )

    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += rand_samples[:, :3]
    root_ang_vel[env_ids] += rand_samples[:, 3:]
    root_lin_vel[env_ids] = torch.where(
        reset_standing_indices.unsqueeze(1),
        sampled_init_root_states_xyzw[:, 7:10],
        root_lin_vel[env_ids]
    )
    root_ang_vel[env_ids] = torch.where(
        reset_standing_indices.unsqueeze(1),
        sampled_init_root_states_xyzw[:, 10:13],
        root_ang_vel[env_ids]
    )

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.where(
        reset_standing_indices.unsqueeze(1),
        sampled_init_dof_pos[:],
        joint_pos[env_ids]
    )
    joint_vel[env_ids] = torch.where(
        reset_standing_indices.unsqueeze(1),
        sampled_init_dof_vel[:],
        joint_vel[env_ids]
    )
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
    )
    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )

    root_state = torch.cat(
      [
        root_pos[env_ids],
        root_ori[env_ids],
        root_lin_vel[env_ids],
        root_ang_vel[env_ids],
      ],
      dim=-1,
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    self.robot.clear_state(env_ids=env_ids)

  def _update_command(self):
    self.time_steps += self._motion_steps_per_control
    env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

    self.prev_anchor_pos[:] = self.current_anchor_pos
    self.current_anchor_pos[:] = self.robot_anchor_pos_w

    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

    if self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

@dataclass(kw_only=True)
class MotionStandingCommandCfg(MotionCommandCfg):
  init_pos_file: str
  root_body_name: tuple[str, ...]
  shoulders_body_names: tuple[str, ...]
  feet_body_names: tuple[str, ...]
  tracking_standing_weight: tuple[float, float] = (1.0, 1.0)

  def build(self, env: ManagerBasedRlEnv) -> MotionStandingCommand:
    return MotionStandingCommand(self, env)
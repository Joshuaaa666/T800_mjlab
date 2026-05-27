"""Offscreen rendering helpers for large-scale training."""

from __future__ import annotations

from typing import Any, Callable

import mujoco

from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer


def patch_offscreen_renderer_for_ghost(env: ManagerBasedRlEnv) -> None:
  """Draw debug visuals after neighbor env geoms so ghost is not occluded.

  The stock ``OffscreenRenderer.update`` calls ``debug_vis_callback`` before
  adding neighboring environment meshes. With many parallel envs those opaque
  copies can hide the semi-transparent ghost. This patch reorders the steps.
  """
  renderer = env._offline_renderer
  if renderer is None:
    return

  def update(
    self,
    data: Any,
    debug_vis_callback: Callable[[MujocoNativeDebugVisualizer], None] | None = None,
    camera: str | None = None,
  ) -> None:
    if self._renderer is None:
      raise ValueError("Renderer not initialized. Call 'initialize()' first.")

    nworld = int(data.nworld)
    if nworld <= 0:
      return

    env_idx = max(0, min(int(self._cfg.env_idx), nworld - 1))
    if self._model.nq > 0:
      self._data.qpos[:] = data.qpos[env_idx].cpu().numpy()
      self._data.qvel[:] = data.qvel[env_idx].cpu().numpy()
    if self._model.nmocap > 0:
      self._data.mocap_pos[:] = data.mocap_pos[env_idx].cpu().numpy()
      self._data.mocap_quat[:] = data.mocap_quat[env_idx].cpu().numpy()
    mujoco.mj_forward(self._model, self._data)

    cam = camera if camera is not None else self._cam
    self._renderer.update_scene(self._data, camera=cam)

    for i in self._get_extra_env_ids(nworld, env_idx):
      if self._model.nq > 0:
        self._data.qpos[:] = data.qpos[i].cpu().numpy()
        self._data.qvel[:] = data.qvel[i].cpu().numpy()
      if self._model.nmocap > 0:
        self._data.mocap_pos[:] = data.mocap_pos[i].cpu().numpy()
        self._data.mocap_quat[:] = data.mocap_quat[i].cpu().numpy()
      mujoco.mj_forward(self._model, self._data)
      mujoco.mjv_addGeoms(
        self._model,
        self._data,
        self._opt,
        self._pert,
        self._catmask.value,
        self._renderer.scene,
      )

    if debug_vis_callback is not None:
      visualizer = MujocoNativeDebugVisualizer(
        self._renderer.scene, self._model, env_idx=self._cfg.env_idx
      )
      debug_vis_callback(visualizer)

  import types

  renderer.update = types.MethodType(update, renderer)

#!/usr/bin/env python3
"""Test T800 robot config by loading t800.py directly."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
MJLAB_SRC = REPO_ROOT / "unitree_rl_mjlab" / "src"
T800_PY = MJLAB_SRC / "assets" / "robots" / "t800" / "t800.py"

if str(MJLAB_SRC) not in sys.path:
  sys.path.insert(0, str(MJLAB_SRC))


def load_t800_module() -> ModuleType:
  """Import t800.py by file path (not via assets.robots.t800 package)."""
  if not T800_PY.exists():
    raise FileNotFoundError(f"t800.py not found: {T800_PY}")
  spec = importlib.util.spec_from_file_location("t800", T800_PY)
  if spec is None or spec.loader is None:
    raise ImportError(f"cannot load module from {T800_PY}")
  module = importlib.util.module_from_spec(spec)
  sys.modules["t800"] = module
  spec.loader.exec_module(module)
  return module


t800 = load_t800_module()

from mjlab.entity.entity import Entity  # noqa: E402

EXPECTED_ACTUATED_JOINTS = (
  "J00_HIP_PITCH_L",
  "J01_HIP_ROLL_L",
  "J02_HIP_YAW_L",
  "J03_KNEE_PITCH_L",
  "J04_ANKLE_PITCH_L",
  "J05_ANKLE_ROLL_L",
  "J06_HIP_PITCH_R",
  "J07_HIP_ROLL_R",
  "J08_HIP_YAW_R",
  "J09_KNEE_PITCH_R",
  "J10_ANKLE_PITCH_R",
  "J11_ANKLE_ROLL_R",
  "J12_TORSO_YAW",
  "J13_SHOULDER_PITCH_L",
  "J14_SHOULDER_ROLL_L",
  "J15_SHOULDER_YAW_L",
  "J16_ELBOW_PITCH_L",
  "J17_ELBOW_YAW_L",
  "J18_SHOULDER_PITCH_R",
  "J19_SHOULDER_ROLL_R",
  "J20_SHOULDER_YAW_R",
  "J21_ELBOW_PITCH_R",
  "J22_ELBOW_YAW_R",
  "J23_HEAD_PITCH",
  "J24_HEAD_YAW",
)


def _joint_names(model: mujoco.MjModel) -> list[str]:
  return [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
    for i in range(model.njnt)
  ]


def _body_names(model: mujoco.MjModel) -> list[str]:
  return [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
    for i in range(model.nbody)
  ]


def test_xml_path() -> None:
  assert t800.T800_XML.exists(), f"XML not found: {t800.T800_XML}"
  print(f"[ok] XML exists: {t800.T800_XML}")


def test_raw_xml_compiles() -> mujoco.MjModel:
  model = mujoco.MjModel.from_xml_path(str(t800.T800_XML))
  assert model.njnt == 26, f"expected 26 joints (incl. free), got {model.njnt}"
  print(f"[ok] raw XML compiles: njnt={model.njnt}, nbody={model.nbody}")
  return model


def test_entity_config() -> tuple[Entity, mujoco.MjModel]:
  robot = Entity(t800.get_t800_robot_cfg())
  model = robot.spec.compile()

  assert model.nu == len(EXPECTED_ACTUATED_JOINTS), (
    f"expected {len(EXPECTED_ACTUATED_JOINTS)} actuators, got {model.nu}"
  )
  assert len(robot.actuator_names) == len(EXPECTED_ACTUATED_JOINTS)
  assert set(robot.actuator_names) == set(EXPECTED_ACTUATED_JOINTS)

  joint_names = set(_joint_names(model)) - {None}
  for name in EXPECTED_ACTUATED_JOINTS:
    assert name in joint_names, f"missing joint: {name}"

  print(f"[ok] Entity config: nu={model.nu}, nbody={model.nbody}")
  print(f"     actuators: {robot.actuator_names[:3]} ... ({len(robot.actuator_names)} total)")
  return robot, model


def test_action_constants() -> None:
  assert len(t800.T800_ACTION_SCALE) == len(EXPECTED_ACTUATED_JOINTS)
  assert len(t800.T800_ACTION_OFFSET) == len(EXPECTED_ACTUATED_JOINTS)
  assert set(t800.T800_ACTION_SCALE) == set(EXPECTED_ACTUATED_JOINTS)
  assert set(t800.T800_ACTION_OFFSET) == set(EXPECTED_ACTUATED_JOINTS)
  for name in EXPECTED_ACTUATED_JOINTS:
    assert t800.T800_ACTION_SCALE[name] > 0

  print(f"[ok] T800_ACTION_SCALE / T800_ACTION_OFFSET: {len(t800.T800_ACTION_SCALE)} joints")


# URDF/motion naming vs MJCF body name in serial_links.xml
_BODY_NAME_ALIASES = {"LINK_TORSO_YAW": "LINK_WAIST_YAW"}


def test_body_names(model: mujoco.MjModel) -> None:
  bodies_in_model = set(_body_names(model))
  missing = []
  for name in t800.T800_BODY_NAMES:
    resolved = _BODY_NAME_ALIASES.get(name, name)
    if resolved not in bodies_in_model:
      missing.append(name)
  assert not missing, f"T800_BODY_NAMES not in model: {missing[:5]}"
  assert len(t800.T800_BODY_NAMES) == 30
  print(f"[ok] T800_BODY_NAMES: {len(t800.T800_BODY_NAMES)} bodies listed")


def test_home_keyframe_defaults() -> None:
  pos = t800.HOME_KEYFRAME.joint_pos
  checks = {
    "J00_HIP_PITCH_L": t800.DEFAULT_Q_HIP_PITCH,
    "J03_KNEE_PITCH_L": t800.DEFAULT_Q_KNEE_PITCH,
    "J14_SHOULDER_ROLL_L": t800.DEFAULT_Q_SHOULDER_ROLL_L,
    "J19_SHOULDER_ROLL_R": t800.DEFAULT_Q_SHOULDER_ROLL_R,
    "J16_ELBOW_PITCH_L": t800.DEFAULT_Q_ELBOW_PITCH,
  }
  for joint, expected in checks.items():
    assert pos[joint] == expected, f"{joint}: {pos[joint]} != {expected}"
  assert t800.HOME_KEYFRAME.pos == (0, 0, 0.8)
  print("[ok] HOME_KEYFRAME joint defaults")


def test_no_duplicate_xml_motors() -> None:
  spec = t800.get_spec()
  assert len(spec.actuators) == 0, (
    "get_spec() should remove XML <motor> actuators before mjlab adds position actuators"
  )
  print("[ok] get_spec() removed XML motor actuators")


def run_tests() -> None:
  print("=== T800 config tests ===")
  print(f"module: {t800.__file__}\n")
  test_xml_path()
  raw_model = test_raw_xml_compiles()
  test_no_duplicate_xml_motors()
  robot, model = test_entity_config()
  del robot
  test_action_constants()
  test_body_names(model)
  test_home_keyframe_defaults()
  del raw_model
  print("\n=== All tests passed ===")


def launch_viewer() -> None:
  robot = Entity(t800.get_t800_robot_cfg())
  import mujoco.viewer as viewer

  print("Launching MuJoCo viewer (close window to exit)...")
  viewer.launch(robot.spec.compile())


def main() -> None:
  parser = argparse.ArgumentParser(description="Test T800 robot XML and mjlab config.")
  parser.add_argument(
    "--viewer",
    action="store_true",
    help="Open MuJoCo viewer after tests pass.",
  )
  args = parser.parse_args()

  run_tests()
  if args.viewer:
    launch_viewer()


if __name__ == "__main__":
  main()

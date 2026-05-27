"""T800 humanoid constants for mjlab (serial_t800.xml)."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# MJCF and assets.
##

T800_XML: Path = SRC_PATH / "assets" / "robots" / "t800" / "xml" / "serial_t800.xml"
assert T800_XML.exists()

# Rigid-body order for serial_t800 (30 links). Motion .npz body_* arrays must follow this list.
T800_BODY_NAMES = [
  "LINK_BASE",
  "LINK_HIP_PITCH_L",
  "LINK_HIP_ROLL_L",
  "LINK_HIP_YAW_L",
  "LINK_KNEE_PITCH_L",
  "LINK_ANKLE_PITCH_L",
  "LINK_ANKLE_ROLL_L",
  "LINK_FOOT_L",
  "LINK_HIP_PITCH_R",
  "LINK_HIP_ROLL_R",
  "LINK_HIP_YAW_R",
  "LINK_KNEE_PITCH_R",
  "LINK_ANKLE_PITCH_R",
  "LINK_ANKLE_ROLL_R",
  "LINK_FOOT_R",
  "LINK_WAIST_YAW",
  "LINK_SHOULDER_PITCH_L",
  "LINK_SHOULDER_ROLL_L",
  "LINK_SHOULDER_YAW_L",
  "LINK_ELBOW_PITCH_L",
  "LINK_ELBOW_YAW_L",
  "LINK_WRIST_END_L",
  "LINK_SHOULDER_PITCH_R",
  "LINK_SHOULDER_ROLL_R",
  "LINK_SHOULDER_YAW_R",
  "LINK_ELBOW_PITCH_R",
  "LINK_ELBOW_YAW_R",
  "LINK_WRIST_END_R",
  "LINK_HEAD_PITCH",
  "LINK_HEAD_YAW",
]

# Subset used by motion-tracking command (must match robot body names in MJCF).
T800_TRACKING_BODY_NAMES = (
  "LINK_BASE",
  "LINK_HIP_ROLL_L",
  "LINK_KNEE_PITCH_L",
  "LINK_ANKLE_ROLL_L",
  "LINK_HIP_ROLL_R",
  "LINK_KNEE_PITCH_R",
  "LINK_ANKLE_ROLL_R",
  "LINK_WAIST_YAW",
  "LINK_SHOULDER_ROLL_L",
  "LINK_ELBOW_PITCH_L",
  "LINK_WRIST_END_L",
  "LINK_SHOULDER_ROLL_R",
  "LINK_ELBOW_PITCH_R",
  "LINK_WRIST_END_R",
)
T800_ANCHOR_BODY_NAME = "LINK_BASE"
T800_EE_BODY_NAMES = (
  "LINK_ANKLE_ROLL_L",
  "LINK_ANKLE_ROLL_R",
  "LINK_WRIST_END_L",
  "LINK_WRIST_END_R",
)

##
# Motor / actuator parameters (aligned with t800参考.py and serial_t800.xml).
##

ARMATURE_Q300H_L = 0.2427264
ARMATURE_Q300H = 0.14110848
ARMATURE_Q200H = 0.0448737
ARMATURE_Q50H = 0.0354625
ARMATURE_Q25H = 0.00671625

EFFORT_LIMIT_Q300H_L = 415.0
EFFORT_LIMIT_Q300H = 370.0
EFFORT_LIMIT_Q200H = 222.0
EFFORT_LIMIT_Q50H = 160.0
EFFORT_LIMIT_Q25H = 52.0

STIFFNESS_Q300H_L = 180.0
STIFFNESS_Q300H = 100.0
STIFFNESS_Q200H = 100.0
STIFFNESS_Q50H = 40.0
STIFFNESS_Q25H = 50.0

DAMPING_Q300H_L = 5.0
DAMPING_Q300H = 3.0
DAMPING_Q200H = 3.0
DAMPING_Q50H = 0.3
DAMPING_Q25H = 0.3

DEFAULT_Q_HIP_PITCH = -0.06
DEFAULT_Q_HIP_ROLL = 0.0
DEFAULT_Q_HIP_YAW = 0.0
DEFAULT_Q_KNEE_PITCH = 0.12
DEFAULT_Q_ANKLE_PITCH = -0.06
DEFAULT_Q_ANKLE_ROLL = 0.0
DEFAULT_Q_TORSO_YAW = 0.0
DEFAULT_Q_SHOULDER_PITCH = 0.0
DEFAULT_Q_SHOULDER_ROLL_L = 0.15
DEFAULT_Q_SHOULDER_ROLL_R = -0.15
DEFAULT_Q_SHOULDER_YAW = 0.0
DEFAULT_Q_ELBOW_PITCH = -0.25
DEFAULT_Q_ELBOW_YAW = 0.0
DEFAULT_Q_HEAD_PITCH = 0.0
DEFAULT_Q_HEAD_YAW = 0.0


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(T800_XML))
  # serial_actuators.xml defines <motor> actuators; replace with mjlab position actuators.
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  return spec


##
# Actuator config.
##

T800_ACTUATOR_Q300H_L = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_HIP_PITCH.*", ".*_KNEE_PITCH.*"),
  stiffness=STIFFNESS_Q300H_L,
  damping=DAMPING_Q300H_L,
  effort_limit=EFFORT_LIMIT_Q300H_L,
  armature=ARMATURE_Q300H_L,
)
T800_ACTUATOR_Q300H = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_HIP_ROLL.*",),
  stiffness=STIFFNESS_Q300H,
  damping=DAMPING_Q300H,
  effort_limit=EFFORT_LIMIT_Q300H,
  armature=ARMATURE_Q300H,
)
T800_ACTUATOR_Q200H = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_HIP_YAW.*",),
  stiffness=STIFFNESS_Q200H,
  damping=DAMPING_Q200H,
  effort_limit=EFFORT_LIMIT_Q200H,
  armature=ARMATURE_Q200H,
)
T800_ACTUATOR_FEET = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ANKLE_PITCH.*", ".*_ANKLE_ROLL.*"),
  stiffness=STIFFNESS_Q50H,
  damping=DAMPING_Q50H,
  effort_limit=EFFORT_LIMIT_Q50H,
  armature=ARMATURE_Q50H,
)
T800_ACTUATOR_TORSO = BuiltinPositionActuatorCfg(
  target_names_expr=("J12_TORSO_YAW",),
  stiffness=STIFFNESS_Q200H,
  damping=DAMPING_Q200H,
  effort_limit=EFFORT_LIMIT_Q200H,
  armature=ARMATURE_Q200H,
)
T800_ACTUATOR_ARMS = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_SHOULDER_PITCH.*",
    ".*_SHOULDER_ROLL.*",
    ".*_SHOULDER_YAW.*",
    ".*_ELBOW_PITCH.*",
  ),
  stiffness=STIFFNESS_Q50H,
  damping=DAMPING_Q50H,
  effort_limit=EFFORT_LIMIT_Q50H,
  armature=ARMATURE_Q50H,
)
T800_ACTUATOR_Q25H = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ELBOW_YAW.*", "J23_HEAD_PITCH", "J24_HEAD_YAW"),
  stiffness=STIFFNESS_Q25H,
  damping=DAMPING_Q25H,
  effort_limit=EFFORT_LIMIT_Q25H,
  armature=ARMATURE_Q25H,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.8),
  joint_pos={
    "J00_HIP_PITCH_L": DEFAULT_Q_HIP_PITCH,
    "J01_HIP_ROLL_L": DEFAULT_Q_HIP_ROLL,
    "J02_HIP_YAW_L": DEFAULT_Q_HIP_YAW,
    "J03_KNEE_PITCH_L": DEFAULT_Q_KNEE_PITCH,
    "J04_ANKLE_PITCH_L": DEFAULT_Q_ANKLE_PITCH,
    "J05_ANKLE_ROLL_L": DEFAULT_Q_ANKLE_ROLL,
    "J06_HIP_PITCH_R": DEFAULT_Q_HIP_PITCH,
    "J07_HIP_ROLL_R": DEFAULT_Q_HIP_ROLL,
    "J08_HIP_YAW_R": DEFAULT_Q_HIP_YAW,
    "J09_KNEE_PITCH_R": DEFAULT_Q_KNEE_PITCH,
    "J10_ANKLE_PITCH_R": DEFAULT_Q_ANKLE_PITCH,
    "J11_ANKLE_ROLL_R": DEFAULT_Q_ANKLE_ROLL,
    "J12_TORSO_YAW": DEFAULT_Q_TORSO_YAW,
    "J13_SHOULDER_PITCH_L": DEFAULT_Q_SHOULDER_PITCH,
    "J14_SHOULDER_ROLL_L": DEFAULT_Q_SHOULDER_ROLL_L,
    "J15_SHOULDER_YAW_L": DEFAULT_Q_SHOULDER_YAW,
    "J16_ELBOW_PITCH_L": DEFAULT_Q_ELBOW_PITCH,
    "J17_ELBOW_YAW_L": DEFAULT_Q_ELBOW_YAW,
    "J18_SHOULDER_PITCH_R": DEFAULT_Q_SHOULDER_PITCH,
    "J19_SHOULDER_ROLL_R": DEFAULT_Q_SHOULDER_ROLL_R,
    "J20_SHOULDER_YAW_R": DEFAULT_Q_SHOULDER_YAW,
    "J21_ELBOW_PITCH_R": DEFAULT_Q_ELBOW_PITCH,
    "J22_ELBOW_YAW_R": DEFAULT_Q_ELBOW_YAW,
    "J23_HEAD_PITCH": DEFAULT_Q_HEAD_PITCH,
    "J24_HEAD_YAW": DEFAULT_Q_HEAD_YAW,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.78),
  joint_pos={
    ".*_HIP_PITCH.*": -0.312,
    ".*_KNEE_PITCH.*": 0.669,
    ".*_ANKLE_PITCH.*": -0.363,
    ".*_ELBOW_PITCH.*": -0.6,
    "J14_SHOULDER_ROLL_L": 0.2,
    "J13_SHOULDER_PITCH_L": 0.2,
    "J19_SHOULDER_ROLL_R": -0.2,
    "J18_SHOULDER_PITCH_R": 0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Final config.
##

T800_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    T800_ACTUATOR_Q300H_L,
    T800_ACTUATOR_Q300H,
    T800_ACTUATOR_Q200H,
    T800_ACTUATOR_FEET,
    T800_ACTUATOR_TORSO,
    T800_ACTUATOR_ARMS,
    T800_ACTUATOR_Q25H,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_t800_robot_cfg() -> EntityCfg:
  """Get a fresh T800 robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    spec_fn=get_spec,
    articulation=T800_ARTICULATION,
  )


T800_ACTION_SCALE: dict[str, float] = {
  "J00_HIP_PITCH_L": 0.5,
  "J01_HIP_ROLL_L": 0.2,
  "J02_HIP_YAW_L": 0.2,
  "J03_KNEE_PITCH_L": 0.5,
  "J04_ANKLE_PITCH_L": 0.5,
  "J05_ANKLE_ROLL_L": 0.2,
  "J06_HIP_PITCH_R": 0.5,
  "J07_HIP_ROLL_R": 0.2,
  "J08_HIP_YAW_R": 0.2,
  "J09_KNEE_PITCH_R": 0.5,
  "J10_ANKLE_PITCH_R": 0.5,
  "J11_ANKLE_ROLL_R": 0.2,
  "J12_TORSO_YAW": 0.2,
  "J13_SHOULDER_PITCH_L": 0.2,
  "J14_SHOULDER_ROLL_L": 0.2,
  "J15_SHOULDER_YAW_L": 0.05,
  "J16_ELBOW_PITCH_L": 0.2,
  "J17_ELBOW_YAW_L": 0.05,
  "J18_SHOULDER_PITCH_R": 0.2,
  "J19_SHOULDER_ROLL_R": 0.2,
  "J20_SHOULDER_YAW_R": 0.05,
  "J21_ELBOW_PITCH_R": 0.2,
  "J22_ELBOW_YAW_R": 0.05,
  "J23_HEAD_PITCH": 0.2,
  "J24_HEAD_YAW": 0.2,
}

T800_ACTION_OFFSET: dict[str, float] = {
  "J00_HIP_PITCH_L": DEFAULT_Q_HIP_PITCH,
  "J01_HIP_ROLL_L": DEFAULT_Q_HIP_ROLL,
  "J02_HIP_YAW_L": DEFAULT_Q_HIP_YAW,
  "J03_KNEE_PITCH_L": DEFAULT_Q_KNEE_PITCH,
  "J04_ANKLE_PITCH_L": DEFAULT_Q_ANKLE_PITCH,
  "J05_ANKLE_ROLL_L": DEFAULT_Q_ANKLE_ROLL,
  "J06_HIP_PITCH_R": DEFAULT_Q_HIP_PITCH,
  "J07_HIP_ROLL_R": DEFAULT_Q_HIP_ROLL,
  "J08_HIP_YAW_R": DEFAULT_Q_HIP_YAW,
  "J09_KNEE_PITCH_R": DEFAULT_Q_KNEE_PITCH,
  "J10_ANKLE_PITCH_R": DEFAULT_Q_ANKLE_PITCH,
  "J11_ANKLE_ROLL_R": DEFAULT_Q_ANKLE_ROLL,
  "J12_TORSO_YAW": DEFAULT_Q_TORSO_YAW,
  "J13_SHOULDER_PITCH_L": DEFAULT_Q_SHOULDER_PITCH,
  "J14_SHOULDER_ROLL_L": DEFAULT_Q_SHOULDER_ROLL_L,
  "J15_SHOULDER_YAW_L": DEFAULT_Q_SHOULDER_YAW,
  "J16_ELBOW_PITCH_L": DEFAULT_Q_ELBOW_PITCH,
  "J17_ELBOW_YAW_L": DEFAULT_Q_ELBOW_YAW,
  "J18_SHOULDER_PITCH_R": DEFAULT_Q_SHOULDER_PITCH,
  "J19_SHOULDER_ROLL_R": DEFAULT_Q_SHOULDER_ROLL_R,
  "J20_SHOULDER_YAW_R": DEFAULT_Q_SHOULDER_YAW,
  "J21_ELBOW_PITCH_R": DEFAULT_Q_ELBOW_PITCH,
  "J22_ELBOW_YAW_R": DEFAULT_Q_ELBOW_YAW,
  "J23_HEAD_PITCH": DEFAULT_Q_HEAD_PITCH,
  "J24_HEAD_YAW": DEFAULT_Q_HEAD_YAW,
}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_t800_robot_cfg())
  viewer.launch(robot.spec.compile())

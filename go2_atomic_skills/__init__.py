"""go2_atomic_skills — two trained Go2 policies as reusable atomic skills.

WalkSkill  a joystick velocity walker (47-d proprioception, stock SB3 PPO).
JumpSkill  a reach-avoid gap-jumping arm (1175-d = 235x5 history, ReachAvoidPPO1P).

Runtime deps: numpy, torch, and (only for the MuJoCo bridge) mujoco. No mjlab,
no safety_sb3.
"""

from .obs import (ACTION_SCALE, BASE_SPAWN_Z, CONTROL_DT, CTRL_GAIN,
                  DEFAULT_JOINT_POS, MJLAB_JOINT_ORDER, ObsInputs,
                  projected_gravity_b)
from .skills import (Go2Skills, Go2ValueFilter, JumpSkill, WalkSkill,
                     flat_ground_scan)
from .mujoco_helper import (fake_gap_scan, obs_from_mujoco,
                            raycast_height_scan)

__all__ = [
  "Go2Skills", "Go2ValueFilter", "WalkSkill", "JumpSkill", "ObsInputs",
  "flat_ground_scan",
  "projected_gravity_b", "MJLAB_JOINT_ORDER", "DEFAULT_JOINT_POS",
  "ACTION_SCALE", "CTRL_GAIN", "CONTROL_DT", "BASE_SPAWN_Z",
  "fake_gap_scan", "obs_from_mujoco", "raycast_height_scan",
]

__version__ = "0.4.1"   # 0.4.1 adds actuator effort limits (forcerange) to the MuJoCo actuator builders; see docs/PATCHES.md

"""Minimal, no-simulator demo: load the skills, build obs from a dummy standing
state, and print valid 12-d actions for both. Needs only numpy + torch."""

from __future__ import annotations

import numpy as np

from go2_atomic_skills import Go2Skills, ObsInputs, DEFAULT_JOINT_POS
from go2_atomic_skills.mujoco_helper import fake_gap_scan

skills = Go2Skills(device="cpu")
skills.reset()

inp = ObsInputs(
  joint_pos=DEFAULT_JOINT_POS.copy(),
  joint_vel=np.zeros(12, dtype=np.float32),
  base_ang_vel=np.zeros(3, dtype=np.float32),
  base_z=0.32,
  base_quat=np.array([1, 0, 0, 0], dtype=np.float32),
)

tgt_walk = skills.walk(inp, command=(1.0, 0.0, 0.0))
print("WALK  joint targets  :", np.round(tgt_walk, 3))
print("      ctrl (post-gain):", np.round(skills.walk_skill.last_action, 3))  # 3.0*a

scan = fake_gap_scan(dist_to_gap=0.35, gap_width=0.30, base_z=0.32)
tgt_jump = skills.jump(inp, command=(1.0, 0.0, 0.0), height_scan=scan)
print("JUMP  joint targets  :", np.round(tgt_jump, 3))
print("      ctrl (post-gain):", np.round(skills.jump_skill.last_action, 3))

assert tgt_walk.shape == (12,) and np.all(np.isfinite(tgt_walk))
assert tgt_jump.shape == (12,) and np.all(np.isfinite(tgt_jump))
print("\nOK — both skills produced finite 12-d joint targets.")

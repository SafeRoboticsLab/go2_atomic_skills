"""End-to-end demo: drive the two atomic skills in a tiny MuJoCo Go2 scene.

Builds a standalone scene from the bundled MJCF (robot + floor + PD position
actuators matching the trained gains, implicitfast integrator), then:
  1. runs WalkSkill closed-loop under a joystick command and prints the REALIZED
     forward speed (should reach ~0.88 m/s at command 1.0), and
  2. runs JumpSkill with a FAKED forward gap in the height-scan (the way to
     invoke the jump with no real terrain),
so the package is demonstrably usable end-to-end.

Run:  python example/demo.py      (needs: numpy, torch, mujoco)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco

from go2_atomic_skills import Go2Skills, MJLAB_JOINT_ORDER
from go2_atomic_skills.mujoco_helper import (build_go2_mjcf_model, set_default_pose,
                                             obs_from_mujoco, fake_gap_scan,
                                             raycast_height_scan)


def fwd_speed(model, data):
  """Body-frame forward velocity (x-component of R^T v_world)."""
  bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
  R = data.xmat[bid].reshape(3, 3)
  return float((R.T @ data.qvel[:3])[0])


def run_walk(model, data, skills, vx, n=400):
  set_default_pose(model, data, base_z=0.33)
  skills.reset()
  print(f"\n--- WALK cmd=({vx},0,0), {n} steps @ 50 Hz ---")
  speeds = []
  for k in range(n):
    inp = obs_from_mujoco(model, data)
    data.ctrl[:] = skills.walk(inp, command=(vx, 0.0, 0.0))
    for _ in range(4):
      mujoco.mj_step(model, data)
    if k >= n - 200:
      speeds.append(fwd_speed(model, data))
    if k % 100 == 0:
      print(f"  step {k:3d}: base_z={float(data.qpos[2]):.3f} fwd={fwd_speed(model,data):.3f}")
  print(f"  realized forward speed (last 200 steps) = {np.mean(speeds):.3f} m/s")
  return float(np.mean(speeds))


def run_jump(model, data, skills, n=60):
  set_default_pose(model, data, base_z=0.33)
  skills.reset()
  print(f"\n--- JUMP faked gap @0.35m w=0.30m, {n} steps ---")
  for k in range(n):
    inp = obs_from_mujoco(model, data)
    scan = fake_gap_scan(dist_to_gap=0.35, gap_width=0.30, base_z=inp.base_z)
    data.ctrl[:] = skills.jump(inp, command=(1.0, 0.0, 0.0), height_scan=scan)
    for _ in range(4):
      mujoco.mj_step(model, data)
    if k % 20 == 0:
      print(f"  step {k:3d}: base_z={float(data.qpos[2]):.3f} "
            f"FL_thigh_target={data.ctrl[1]:.3f}")


def main():
  model = build_go2_mjcf_model()          # floor + PD actuators + implicitfast
  data = mujoco.MjData(model)
  skills = Go2Skills(device="cpu")

  run_walk(model, data, skills, vx=1.0)
  run_jump(model, data, skills)

  # real down-raycast over the flat floor (mask to the floor's geom group 0)
  set_default_pose(model, data, base_z=0.33)
  scan = raycast_height_scan(model, data, geomgroup=np.array([1, 0, 0, 0, 0, 0], np.uint8))
  print(f"\nraycast_height_scan over flat floor: min={scan.min():.3f} "
        f"max={scan.max():.3f} (expect ~base_z, no gap)")
  print("\nDONE.")


if __name__ == "__main__":
  main()

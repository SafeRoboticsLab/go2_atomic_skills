"""(b) FAKE-GAP-SCAN PARITY for the E098 handover arm (tag 'hando_w30').

Does the synthetic `fake_gap_scan` override reproduce what the E098 arm would
see (and decide) from a REAL height-field gap? For a sweep of dist_to_gap and
two gap widths, we build a MuJoCo scene with a real trench (near platform / deep
gap floor / far platform), raycast the real 187-ray scan, and compare the arm's
V(s) and mean action on the REAL scan vs the `fake_gap_scan` for the same
physical state (identical proprioception; only the scan differs).

CPU-only (package MuJoCo + torch). Reports per-distance max|ΔV|, V-sign
agreement, max|Δa|.
"""
from __future__ import annotations
import sys
import numpy as np

PKG = "/home/buzi/Desktop/RESEARCH/SAFE/DEVELOPMENT/go2_atomic_skills"
sys.path.insert(0, PKG)
import mujoco

from go2_atomic_skills.obs import DEFAULT_JOINT_POS, MJLAB_JOINT_ORDER, SCAN_MAX_DISTANCE
from go2_atomic_skills.mujoco_helper import (MJCF_PATH, _GAINS, fake_gap_scan,
                                             obs_from_mujoco, raycast_height_scan)
from go2_atomic_skills.skills import JumpSkill

BASE_Z = 0.33
GAP_DEPTH = 1.0
TERRAIN_GROUP = 3


def build_gap_model(dist_to_gap, gap_width):
  """Go2 + a real trench: near platform (top z=0) up to x=dist_to_gap, deep gap
  floor (top z=-GAP_DEPTH) across [dist_to_gap, dist_to_gap+gap_width], far
  platform (top z=0) beyond. Terrain geoms in group TERRAIN_GROUP so the raycast
  mask isolates them from the robot's own legs. Robot base sits at x=0."""
  spec = mujoco.MjSpec.from_file(MJCF_PATH)
  wb = spec.worldbody

  def box(name, xc, zc, hx, hz):
    g = wb.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [hx, 2.0, hz]
    g.pos = [xc, 0.0, zc]
    g.group = TERRAIN_GROUP
    g.rgba = [0.4, 0.5, 0.6, 1]

  near_end = dist_to_gap
  far_start = dist_to_gap + gap_width
  # near platform: x in [-3, near_end], top at 0
  box("near", (-3.0 + near_end) / 2, -0.5, (near_end + 3.0) / 2, 0.5)
  # far platform: x in [far_start, far_start+3], top at 0
  box("far", (far_start + far_start + 3.0) / 2, -0.5, 1.5, 0.5)
  # deep gap floor: top at -GAP_DEPTH, spanning the whole scan x-range under gap
  box("deep", (near_end + far_start) / 2, -GAP_DEPTH - 0.5, gap_width / 2, 0.5)

  for jname in MJLAB_JOINT_ORDER:
    grp = "hip" if "hip" in jname else "thigh" if "thigh" in jname else "calf"
    spec.joint(jname).armature = _GAINS[grp][2]
  m = spec.compile()
  m.opt.timestep = 0.005
  m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  return m


def standing_state(m):
  d = mujoco.MjData(m)
  d.qpos[:3] = [0.0, 0.0, BASE_Z]
  d.qpos[3:7] = [1, 0, 0, 0]
  for i, nm in enumerate(MJLAB_JOINT_ORDER):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
    d.qpos[int(m.jnt_qposadr[jid])] = DEFAULT_JOINT_POS[i]
  d.qvel[:] = 0
  mujoco.mj_forward(m, d)
  return d


def eval_arm(scan, base_z, command=(1.0, 0.0, 0.0)):
  """V and mean action from a fresh E098 arm, backfilled with this single frame
  (history = 5x the frame), one step. Returns (V, action[12])."""
  import torch
  from go2_atomic_skills.obs import ObsInputs
  inp = ObsInputs(joint_pos=DEFAULT_JOINT_POS.copy(), joint_vel=np.zeros(12),
                  base_ang_vel=np.zeros(3), base_z=base_z,
                  base_quat=np.array([1, 0, 0, 0.0]))
  sk = JumpSkill(device="cpu", width="hando_w30", with_critic=True)
  sk.reset()
  v, ctrl = sk.value_and_action(inp, command, scan)
  a = (ctrl / 3.0)  # back out raw action (unclipped) for reporting
  return v, a


def main():
  gmask = np.zeros(6, dtype=np.uint8)
  gmask[TERRAIN_GROUP] = 1
  dists = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
  print("=== (b) FAKE-GAP-SCAN PARITY — E098 hando arm ===")
  for gap_w in (0.30, 0.20):
    print(f"\n--- gap_width = {gap_w:.2f} m ---")
    print(f"{'dist':>5} {'V_real':>8} {'V_fake':>8} {'|dV|':>8} "
          f"{'signOK':>6} {'|da|':>7} {'scan|d|':>8}")
    for dg in dists:
      m = build_gap_model(dg, gap_w)
      d = standing_state(m)
      real = raycast_height_scan(m, d, base_body="base_link", geomgroup=gmask)
      base_z = float(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")][2])
      fake = fake_gap_scan(dg, gap_w, base_z, gap_depth=GAP_DEPTH)
      vr, ar = eval_arm(real, base_z)
      vf, af = eval_arm(fake, base_z)
      sign_ok = (np.sign(vr) == np.sign(vf))
      scan_d = float(np.abs(real - fake).max())
      print(f"{dg:5.2f} {vr:8.4f} {vf:8.4f} {abs(vr-vf):8.4f} "
            f"{str(bool(sign_ok)):>6} {np.abs(ar-af).max():7.4f} {scan_d:8.3f}")


if __name__ == "__main__":
  main()

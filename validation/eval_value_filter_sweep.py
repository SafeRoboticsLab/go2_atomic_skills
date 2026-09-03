"""(c)+(d) VALUE-FILTER behaviour sweep for the deployment arm.

Closed-loop MuJoCo rollouts (CPU) of the walker approaching a gap under
``Go2ValueFilter``, over an eps sweep (value trigger) and a D sweep (distance
trigger), on a FAKE-gap override scene (flat floor + synthetic scan) and a REAL
height-field gap (near platform / deep trench / far platform + real raycast).
Compares the default deployment arm (E098 hando_w30) against the alternate
E040b (w30), and isolates the unclipped-control fix (E040b clipped vs unclipped).

Per cell (>=N_ROLL rollouts): engagement distance, crossing rate, outcome mix,
landing pitch (mean / p90) and head/trunk-first ground-contact fraction (the
head-dive symptom).

Run:  CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl python validation/eval_value_filter_sweep.py [--quick]
"""
from __future__ import annotations
import os, sys, math, argparse
import numpy as np

PKG = "/home/buzi/Desktop/RESEARCH/SAFE/DEVELOPMENT/go2_atomic_skills"
sys.path.insert(0, PKG)
import mujoco
import torch
torch.set_num_threads(2)

from go2_atomic_skills.obs import DEFAULT_JOINT_POS, MJLAB_JOINT_ORDER
from go2_atomic_skills.mujoco_helper import (MJCF_PATH, _EFFORT_LIMITS, _GAINS,
                                             build_go2_mjcf_model,
                                             fake_gap_scan, obs_from_mujoco,
                                             raycast_height_scan)
from go2_atomic_skills.skills import Go2ValueFilter

GAP_DEPTH = 1.0
TERRAIN_GROUP = 4                       # robot collisions are group 3; keep terrain apart
GMASK = np.zeros(6, np.uint8); GMASK[TERRAIN_GROUP] = 1
BASE_Z0 = 0.33
FOOT = tuple(f"{p}_foot_collision" for p in ("FL", "FR", "RL", "RR"))
TRUNK = ("base1_collision", "base2_collision", "base3_collision")


# ---------- scenes ----------
def add_actuators_and_arm(spec):
  import mujoco
  wb = spec.worldbody
  for jname in MJLAB_JOINT_ORDER:
    grp = "hip" if "hip" in jname else "thigh" if "thigh" in jname else "calf"
    kp, kv, arm = _GAINS[grp]
    spec.joint(jname).armature = arm
    a = spec.add_actuator(name=jname, target=jname, trntype=mujoco.mjtTrn.mjTRN_JOINT)
    a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    a.gainprm[0] = kp; a.biasprm[1] = -kp; a.biasprm[2] = -kv
    a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    lim = _EFFORT_LIMITS[grp]; a.forcerange[0] = -lim; a.forcerange[1] = lim


def build_flat():
  m = build_go2_mjcf_model()            # floor plane (group 0) + actuators + implicitfast
  return m


def build_gap(gap_width):
  """Near platform (top z=0, x in [-4,0]); deep gap floor (top -GAP_DEPTH,
  x in [0,gap_width]); far platform (top 0, x in [gap_width, gap_width+4]).
  Terrain geoms in group TERRAIN_GROUP so the raycast mask isolates them."""
  spec = mujoco.MjSpec.from_file(MJCF_PATH)
  wb = spec.worldbody
  def box(name, xc, zc, hx, hz):
    g = wb.add_geom(); g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = [hx, 2.0, hz]; g.pos = [xc, 0, zc]
    g.group = TERRAIN_GROUP; g.rgba = [0.4, 0.5, 0.6, 1]
  box("near", -2.0, -0.5, 2.0, 0.5)
  box("far", gap_width + 2.0, -0.5, 2.0, 0.5)
  box("deep", gap_width / 2, -GAP_DEPTH - 0.5, gap_width / 2, 0.5)
  add_actuators_and_arm(spec)
  m = spec.compile(); m.opt.timestep = 0.005
  m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  return m


# ---------- helpers ----------
def _bid(m): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")

def set_spawn(m, d, x0, vx0, rng):
  jadr = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)])
          for nm in MJLAB_JOINT_ORDER]
  d.qpos[:] = 0
  d.qpos[:3] = [x0, 0.0, BASE_Z0]
  d.qpos[3:7] = [1, 0, 0, 0]
  jitter = rng.normal(0, 0.03, 12).astype(np.float32)
  for i, a in enumerate(jadr):
    d.qpos[a] = DEFAULT_JOINT_POS[i] + jitter[i]
  d.qvel[:] = 0
  d.qvel[0] = vx0
  mujoco.mj_forward(m, d)

def pitch_deg(m, d):
  R = d.xmat[_bid(m)].reshape(3, 3)
  return math.degrees(math.asin(max(-1.0, min(1.0, R[2, 0]))))  # nose-up +, dive -

def geom_body(m, gid): return int(m.geom_bodyid[gid])

def contacts(m, d):
  """Return (trunk_terrain, foot_terrain) booleans this step: is any TRUNK geom
  touching terrain; is any FOOT touching terrain. Terrain = worldbody geoms."""
  trunk = foot = False
  robot_trunk = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in TRUNK}
  robot_foot = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT}
  for i in range(d.ncon):
    c = d.contact[i]; g1, g2 = c.geom1, c.geom2
    b1, b2 = geom_body(m, g1), geom_body(m, g2)
    world = (b1 == 0) or (b2 == 0)
    if not world:
      continue
    rg = g1 if b2 == 0 else g2
    if rg in robot_trunk: trunk = True
    if rg in robot_foot: foot = True
  return trunk, foot


# ---------- one rollout ----------
def rollout(m, filt, *, mode, gap_width, trigger, eps, D, seed, max_steps=360):
  rng = np.random.default_rng(seed)
  x0 = rng.uniform(-1.50, -1.20)
  vx0 = rng.uniform(0.55, 0.80)
  d = mujoco.MjData(m)
  set_spawn(m, d, x0, vx0, rng)
  filt.reset()
  # phase decorrelation: advance both clocks by a random offset
  off = int(rng.integers(0, 30)); filt.walk_skill.step = off; filt.jump_skill.step = off
  bid = _bid(m)
  engaged_dist = float("nan"); ever_engaged = False
  min_z_over_gap = 9.9
  head_first = False
  landing_pitch = float("nan"); landed = False
  crossed_edge = False
  for k in range(max_steps):
    base_x = float(d.xpos[bid][0]); base_z = float(d.xpos[bid][2])
    dist = -base_x                                   # gap near-edge at world x=0
    inp = obs_from_mujoco(m, d)
    if mode == "fake":
      scan = fake_gap_scan(dist, gap_width, inp.base_z, gap_depth=GAP_DEPTH)
      dgap = dist
    else:
      scan = raycast_height_scan(m, d, geomgroup=GMASK)
      dgap = None                                     # let distance-trigger estimate
    tgt, info = filt.step(inp, command=(0.9, 0.0, 0.0), height_scan=scan,
                          trigger=trigger, eps=eps, D=D,
                          dist_to_gap=(dist if trigger == "distance" and mode == "fake" else dgap))
    if info["engaged"] and not ever_engaged:
      ever_engaged = True; engaged_dist = dist
    d.ctrl[:] = tgt
    for _ in range(4):
      mujoco.mj_step(m, d)
    base_x = float(d.xpos[bid][0]); base_z = float(d.xpos[bid][2])
    if -0.10 <= base_x <= gap_width + 0.10:
      min_z_over_gap = min(min_z_over_gap, base_z)
    # head/trunk-first contact within the crossing/landing window
    if -0.05 <= base_x <= gap_width + 0.35:
      tr, ft = contacts(m, d)
      if tr:
        head_first = True
    # landing: first far-side foot contact after passing the far edge
    if not landed and base_x > gap_width:
      _tr, ft = contacts(m, d)
      if ft:
        landed = True; landing_pitch = pitch_deg(m, d)
    if base_x > gap_width:
      crossed_edge = True
    if base_z < -0.35:                                # deep in the trench: stop early
      break
  base_xf = float(d.xpos[bid][0]); base_zf = float(d.xpos[bid][2])
  R = d.xmat[bid].reshape(3, 3); upright = R[2, 2] > 0.5 and base_zf > 0.18
  fell_in = (min_z_over_gap < 0.12)
  if fell_in:
    outcome = "fall"
  elif base_xf > gap_width + 0.20 and upright:
    outcome = "cross"
  elif not upright:
    outcome = "topple"
  else:
    outcome = "short"
  return dict(outcome=outcome, engaged_dist=engaged_dist, ever_engaged=ever_engaged,
              landing_pitch=landing_pitch if outcome == "cross" else float("nan"),
              head_first=head_first if crossed_edge else False,
              base_xf=base_xf, crossed_edge=crossed_edge)


def agg(rs):
  n = len(rs)
  oc = {k: sum(r["outcome"] == k for r in rs) / n for k in ("cross", "fall", "short", "topple")}
  eng = [r["engaged_dist"] for r in rs if r["ever_engaged"] and np.isfinite(r["engaged_dist"])]
  lp = [r["landing_pitch"] for r in rs if np.isfinite(r["landing_pitch"])]
  crossers = [r for r in rs if r["crossed_edge"]]
  hf = (sum(r["head_first"] for r in crossers) / len(crossers)) if crossers else float("nan")
  return dict(
    n=n, cross=oc["cross"], fall=oc["fall"], short=oc["short"], topple=oc["topple"],
    eng_frac=len(eng) / n,
    eng_dist=(float(np.mean(eng)) if eng else float("nan")),
    lp_mean=(float(np.mean(lp)) if lp else float("nan")),
    lp_p90=(float(np.percentile(lp, 90)) if lp else float("nan")),
    head_first=hf)


def row(tag, a):
  print(f"{tag:28s} cross={a['cross']:.2f} fall={a['fall']:.2f} short={a['short']:.2f} "
        f"top={a['topple']:.2f} | engF={a['eng_frac']:.2f} engD={a['eng_dist']:.3f} "
        f"| pitch mean={a['lp_mean']:6.1f} p90={a['lp_p90']:6.1f} headF={a['head_first']:.2f}")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--quick", action="store_true")
  ap.add_argument("--nroll", type=int, default=32)
  ap.add_argument("--gap", type=float, default=0.30)
  args = ap.parse_args()
  N = 8 if args.quick else args.nroll
  gap = args.gap
  eps_list = [-0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
  D_list = [0.30, 0.40, 0.50]
  arms = {"E098(hando_w30)": dict(width="hando_w30", clip=True),
          "E040b(w30)": dict(width="w30", clip=True)}

  flat = build_flat()
  gapm = build_gap(gap)

  def run_cell(width, clip, mode, trigger, eps, D):
    m = flat if mode == "fake" else gapm
    filt = Go2ValueFilter(device="cpu", jump_width=width, eps=eps, trigger=trigger,
                          D=D, clip_action=clip)
    rs = [rollout(m, filt, mode=mode, gap_width=gap, trigger=trigger, eps=eps, D=D,
                  seed=1000 + i) for i in range(N)]
    return agg(rs)

  for mode in ("fake", "real"):
    print(f"\n{'='*96}\n=== MODE={mode.upper()}  gap={gap:.2f}  N={N}  (value trigger: eps sweep) ===")
    for name, cfg in arms.items():
      print(f"-- {name} --")
      for eps in eps_list:
        row(f"  eps={eps:+.2f}", run_cell(cfg["width"], cfg["clip"], mode, "value", eps, 0.40))
    print(f"--- distance trigger: D sweep ---")
    for name, cfg in arms.items():
      print(f"-- {name} --")
      for D in D_list:
        row(f"  D={D:.2f}", run_cell(cfg["width"], cfg["clip"], mode, "distance", 0.25, D))

  # unclip isolation: E040b clipped vs unclipped, value eps=0.25, both modes
  print(f"\n{'='*96}\n=== UNCLIP ISOLATION (E040b w30, value eps=0.25) ===")
  for mode in ("fake", "real"):
    for clip in (True, False):
      tag = f"  {mode} clip={clip}"
      row(tag, run_cell("w30", clip, mode, "value", 0.25, 0.40))


if __name__ == "__main__":
  main()

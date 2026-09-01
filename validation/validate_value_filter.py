"""HONEST faithfulness gate for the packaged VALUE FILTER.

Proves the package's :class:`Go2ValueFilter` reproduces the source repo's eval
harness value filter (``build_filter("value", ...)`` on the RA w30 arm) to
numerical noise — the certificate V(s) AND the least-restrictive engage decision
(``V <= eps``) — on the SAME live sim states.

Design (per-env LOCKSTEP): we run the real harness filter batched over a handful
of envs, and one package ``Go2ValueFilter`` per env driven on the SAME state each
step. Because the harness records the SELECTED (applied) action into BOTH obs
groups' ``actions`` term, the package must set both the walker's AND the jump's
``last_action`` to that selected control every step; if it does not, the jump's
5-frame history diverges and V drifts. That is exactly the "shared applied
last_action" subtlety this test is built to catch — the comparison is on V, so
any bookkeeping error surfaces as a V mismatch, not a silent wrong number.

The package is fed:
  * proprioception STATE via ``obs_from_mujoco`` on a plain MjData transferred
    (by name) from the live mjlab state — so the mujoco bridge is on the hook;
  * the COMMAND and the raw HEIGHT-SCAN read straight out of the env's own
    pre-norm safety obs (the newest history frame), so what the package scales
    and stacks is bit-identical to what the twin saw.

Gate: package V matches harness V to < 1e-4 on every (env, step), and the engage
mask matches EXACTLY. On a done step the env auto-resets; we reset that env's
package filter in lockstep so histories/phase stay aligned.

Run (from the source repo, mjlab env):
  MUJOCO_GL=egl PYTHONPATH=<repo>:<safety-sb3>:<pkg> \
  ~/miniconda3/envs/mjlab/bin/python validation/validate_value_filter.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "/home/buzi/Desktop/RESEARCH/SAFE/DEVELOPMENT/go2_atomic_skills")

import mujoco

from go2_atomic_skills.obs import MJLAB_JOINT_ORDER, SCAN_N, SCAN_SCALE
from go2_atomic_skills.mujoco_helper import build_go2_mjcf_model, obs_from_mujoco
from go2_atomic_skills.skills import Go2ValueFilter

ROOT = "/home/buzi/Desktop/RESEARCH/SAFE/DEVELOPMENT/safe_mjlab_zoo"
WALKER = f"{ROOT}/runs/go2_walker_flat/final_model.zip"
ARM = f"{ROOT}/runs/gap_e040_resaved/ra_w30/final_model.zip"
TASK = "go2_gap_brake_or_jump_ra_w30"

EPS = 0.25
V_TOL = 1e-4
NUM_ENVS = 8
STEPS = 120
DEV = "cuda:0"

# term-major history layout offsets (see go2_atomic_skills.obs.HistoryBuffer):
# newest frame of each term is the LAST `dim` of that term's 5-frame block.
CMD_NEWEST = slice(42, 45)          # command (3), newest frame
SCAN_NEWEST = slice(1175 - SCAN_N, 1175)  # height_scan (187), newest frame


def _np(t):
  return t.detach().cpu().numpy()


def _transfer_state(mjm, mjd, robot, imu, e):
  """Write env ``e``'s live mjlab state into a plain MjData by NAME, mj_forward."""
  d = robot.data
  mjd.qpos[:3] = _np(d.root_link_pos_w[e])
  mjd.qpos[3:7] = _np(d.root_link_quat_w[e])           # w,x,y,z
  mjd.qvel[:3] = _np(d.root_link_lin_vel_w[e])         # free-joint linear = global
  mjd.qvel[3:6] = _np(imu.data[e])                     # body-frame gyro = imu reading
  for i, nm in enumerate(MJLAB_JOINT_ORDER):
    jid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, nm)
    mjd.qpos[int(mjm.jnt_qposadr[jid])] = float(d.joint_pos[e, i])
    mjd.qvel[int(mjm.jnt_dofadr[jid])] = float(d.joint_vel[e, i])
  mujoco.mj_forward(mjm, mjd)


def main():
  from robot_safety_sandbox.eval import (NominalPolicy, SwitchCfg,
                                         build_eval_env, build_filter,
                                         load_nominal, load_twin, safety_modules)
  from robot_safety_sandbox.envs.go2_gap.eval_gauntlet import gap_cfg_transform

  print("=== VALUE-FILTER faithfulness gate ===")
  print(f"  task={TASK}  arm=ra_w30  eps={EPS}  n_envs={NUM_ENVS}  steps={STEPS}")

  # graft the walker's 47-d 'actor' group + a standing walk-in spawn on an
  # extended island, so the nominal reads what it trained on (matches the
  # filter_traj gauntlet the eps=0.25 recipe was tuned on).
  transform = gap_cfg_transform(gap_width=0.30, n_gaps=1, episode_s=20.0,
                                cmd_vx=1.0, island_length=3.0, full_pose=True)
  env = build_eval_env(TASK, NUM_ENVS, DEV, cfg_transform=transform)
  print(f"  obs: nominal='{env.nominal_obs_key}' safety='{env.safety_obs_key}'")

  safety_model, norm = load_twin(ARM, DEV)
  mods = safety_modules(safety_model, env.num_envs, DEV)
  mods["norm"] = norm
  bundle = build_filter("value", mods, env, switch=SwitchCfg(eps=EPS))
  filt = bundle.filt
  nominal = NominalPolicy(*load_nominal(WALKER, DEV), device=DEV)

  # one plain MuJoCo model + reusable data for the state transfer, and one
  # package filter per env, all reset fresh in lockstep with env.reset().
  mjm = build_go2_mjcf_model()
  mjd = mujoco.MjData(mjm)
  pkg = [Go2ValueFilter(device="cpu", jump_width="w30", eps=EPS)
         for _ in range(NUM_ENVS)]
  for f in pkg:
    f.reset()

  env.reset(seed=0)
  robot = env.robot
  imu = env.mj.scene["robot/imu_ang_vel"]
  prev_done = torch.ones(NUM_ENVS, dtype=torch.bool, device=DEV)

  worst_dv = 0.0
  n_cmp = 0
  n_mask_mismatch = 0
  n_engaged_h = 0
  worst_by_env = np.zeros(NUM_ENVS)
  for t in range(STEPS):
    a_nom = nominal(env.nominal_obs())
    s_raw = env.safety_obs()                    # pre-norm proprioception (N,1175)
    s_obs = norm(s_raw)
    speed = torch.norm(env.robot.data.root_link_lin_vel_w[:, :2], dim=1)
    action, info = filt(a_nom, speed=speed, fresh=prev_done, s_obs=s_obs)
    Vh = _np(info.value).astype(np.float64)               # (N,)
    engaged_h = _np(info.engaged).astype(bool)            # (N,)

    s_np = _np(s_raw)
    cmd_np = s_np[:, CMD_NEWEST]                           # (N,3) newest command
    scan_np = s_np[:, SCAN_NEWEST] / SCAN_SCALE            # (N,187) RAW newest scan

    for e in range(NUM_ENVS):
      _transfer_state(mjm, mjd, robot, imu, e)
      inp = obs_from_mujoco(mjm, mjd)
      _tgt, pinfo = pkg[e].step(inp, command=tuple(cmd_np[e]),
                                height_scan=scan_np[e], eps=EPS)
      dv = abs(pinfo["value"] - Vh[e])
      worst_dv = max(worst_dv, dv)
      worst_by_env[e] = max(worst_by_env[e], dv)
      n_cmp += 1
      n_engaged_h += int(engaged_h[e])
      if pinfo["engaged"] != bool(engaged_h[e]):
        n_mask_mismatch += 1
        print(f"  [MASK MISMATCH] t={t} env={e} pkg_engaged={pinfo['engaged']} "
              f"harness={bool(engaged_h[e])}  Vpkg={pinfo['value']:.5f} "
              f"Vh={Vh[e]:.5f}")
      elif dv > V_TOL:
        print(f"  [dV>{V_TOL}] t={t} env={e} dV={dv:.2e}  Vpkg={pinfo['value']:.5f} "
              f"Vh={Vh[e]:.5f}")

    out = env.step(action)
    done = out.done
    prev_done = done.clone()
    if bool(done.any()):
      filt.reset(done)
      for e in np.nonzero(_np(done))[0]:        # keep package aligned with reset
        pkg[int(e)].reset()

  bundle.close()
  env.close()

  print("\n=== RESULTS ===")
  print(f"  comparisons            : {n_cmp}  ({NUM_ENVS} envs x {STEPS} steps)")
  print(f"  harness engaged frac   : {n_engaged_h / max(n_cmp,1):.3f} "
        f"(both branches exercised iff in (0,1))")
  print(f"  worst |V_pkg - V_harness| = {worst_dv:.3e}   (tol {V_TOL:.0e})")
  print(f"  per-env worst |dV|     : "
        + " ".join(f"{x:.1e}" for x in worst_by_env))
  print(f"  engage-mask mismatches : {n_mask_mismatch} / {n_cmp}")
  v_ok = worst_dv < V_TOL
  mask_ok = n_mask_mismatch == 0
  print(f"\n  V gate     : {'PASS' if v_ok else 'FAIL'} "
        f"(worst {worst_dv:.2e} < {V_TOL:.0e})")
  print(f"  mask gate  : {'PASS' if mask_ok else 'FAIL'} "
        f"({n_mask_mismatch} mismatches)")
  ok = v_ok and mask_ok
  print("\n  RESULT:", "ALL PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  np.random.seed(0)
  torch.manual_seed(0)
  sys.exit(main())

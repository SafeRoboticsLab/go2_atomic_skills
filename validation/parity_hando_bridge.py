"""(a) BRIDGE + NET PARITY for the deployment arm (E098, tag 'hando_w30').

Three clean measurements against the source twin, on live mjlab states:
  1. single-frame obs bridge (obs_from_mujoco vs env obs), per term;
  2. feed-forward net+normalizer parity (package actor/critic on the env's own
     raw obs vs the twin's mean action / predict_values);
  3. closed-loop bridge parity — every term matches to ~1e-7 EXCEPT the gait
     phase term, which the env randomizes on reset while the package owns its
     clock (a documented deployment property, not a fidelity gap).

Deployment convention note: the arm's action is CLAMPED to [-1,1] before the
gain in the eval/deployment filter (eval.policies.safety_modules), which is the
convention the composed result was validated under and the package's default;
this script drives the env directly with the raw twin action (base.py path,
unclamped) purely to isolate the bridge + net, so the closed-loop history is
seeded from the env's own actions term.

Run (source repo, mjlab env):
  MUJOCO_GL=egl PYTHONPATH=<repo>:<sb3>:<pkg> \
  ~/miniconda3/envs/mjlab/bin/python validation/parity_hando_bridge.py
"""
from __future__ import annotations
import sys
import numpy as np
import torch

PKG = "/home/buzi/Desktop/RESEARCH/SAFE/DEVELOPMENT/go2_atomic_skills"
sys.path.insert(0, PKG)
import mujoco

from go2_atomic_skills.obs import MJLAB_JOINT_ORDER, SCAN_SCALE
from go2_atomic_skills.mujoco_helper import build_go2_mjcf_model, obs_from_mujoco
from go2_atomic_skills.skills import JumpSkill

ROOT = "/home/buzi/Desktop/RESEARCH/SAFE/DEVELOPMENT/safe_mjlab_zoo"
ARM = f"{ROOT}/runs/ras_e098/go2_gap_brake_or_jump_ra_hando/final_model.zip"
TASK = "go2_gap_brake_or_jump_ra_w30"   # state source; obs contract == hando
TAG = "hando_w30"
NUM_ENVS = 8
STEPS = 120
DEV = "cuda:0"
TERMS = [("bav", 0, 15), ("pg", 15, 30), ("cmd", 30, 45), ("ph", 45, 55),
         ("bh", 55, 60), ("jp", 60, 120), ("jv", 120, 180), ("act", 180, 240),
         ("scan", 240, 1175)]


def _np(t):
  return t.detach().cpu().numpy()


def transfer(mjm, mjd, d, imu, e):
  mjd.qpos[:3] = _np(d.root_link_pos_w[e])
  mjd.qpos[3:7] = _np(d.root_link_quat_w[e])
  mjd.qvel[:3] = _np(d.root_link_lin_vel_w[e])
  mjd.qvel[3:6] = _np(imu.data[e])
  for i, nm in enumerate(MJLAB_JOINT_ORDER):
    jid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, nm)
    mjd.qpos[int(mjm.jnt_qposadr[jid])] = float(d.joint_pos[e, i])
    mjd.qvel[int(mjm.jnt_dofadr[jid])] = float(d.joint_vel[e, i])
  mujoco.mj_forward(mjm, mjd)


def main():
  from robot_safety_sandbox.eval import build_eval_env, load_twin
  print(f"=== (a) BRIDGE + NET PARITY — E098 ({TAG}) ===")
  env = build_eval_env(TASK, NUM_ENVS, DEV)
  model, norm = load_twin(ARM, DEV, quiet=True)
  policy = model.policy
  policy.set_training_mode(False)
  mjm = build_go2_mjcf_model()
  mjd = mujoco.MjData(mjm)
  pk = [JumpSkill(device="cpu", width=TAG, with_critic=True) for _ in range(NUM_ENVS)]
  for p in pk:
    p.reset()
  env.reset(seed=0)
  d, imu = env.robot.data, env.mj.scene["robot/imu_ang_vel"]

  sf = 0.0                     # single-frame bridge worst
  wff_a = wff_v = 0.0          # feed-forward net worst
  termw = {nm: 0.0 for nm, _, _ in TERMS}
  for t in range(STEPS):
    s = _np(env.safety_obs())
    with torch.no_grad():
      a = _np(policy._predict(norm(env.safety_obs()), deterministic=True))
      vt = _np(policy.predict_values(norm(env.safety_obs())).squeeze(-1)).astype(np.float64)
      oo = pk[0].norm(torch.as_tensor(s))
      wff_a = max(wff_a, float(np.abs(_np(pk[0].net(oo)) - a).max()))
      wff_v = max(wff_v, float(np.abs(_np(pk[0].critic(oo).squeeze(-1)) - vt).max()))
    for e in range(NUM_ENVS):
      pk[e].last_action = s[e, 228:240].astype(np.float32).copy()  # env-seeded history
      transfer(mjm, mjd, d, imu, e)
      inp = obs_from_mujoco(mjm, mjd)
      # single-frame bridge check (fresh skill, no history influence except seed)
      frame = pk[e].raw_frame(inp, tuple(s[e, 42:45]), s[e, 988:1175] / SCAN_SCALE)
      pk[e].history.append(frame)
      obs = pk[e].history.flatten()
      if t == 0:
        sf = max(sf, float(np.abs(obs - s[e]).max()))
      for nm, aa, bb in TERMS:
        termw[nm] = max(termw[nm], float(np.abs(obs[aa:bb] - s[e, aa:bb]).max()))
    out = env.step(torch.as_tensor(a, device=DEV))
    for e in np.nonzero(_np(out.done))[0]:   # keep package histories aligned on reset
      pk[int(e)].reset()
  env.close()

  print(f"\n  (1) single-frame bridge worst |Δ| (t=0, all envs): {sf:.3e}")
  print(f"  (2) feed-forward net parity: worst |Δa|={wff_a:.3e}  worst |ΔV|={wff_v:.3e}")
  print("  (3) closed-loop per-term worst |Δ|:")
  for nm, _, _ in TERMS:
    flag = "  <- env-randomized gait phase (deployment-owned; not a fidelity gap)" if nm == "ph" else ""
    print(f"        {nm:5s} {termw[nm]:.3e}{flag}")
  ok = sf < 1e-4 and wff_a < 1e-4 and wff_v < 1e-4 and all(
    termw[nm] < 1e-4 for nm, _, _ in TERMS if nm != "ph")
  print(f"\n  RESULT: {'PASS' if ok else 'FAIL'} (all non-phase terms < 1e-4)")
  return 0 if ok else 1


if __name__ == "__main__":
  np.random.seed(0)
  torch.manual_seed(0)
  sys.exit(main())

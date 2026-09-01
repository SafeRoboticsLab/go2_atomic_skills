"""CORRECTNESS GATE: hand-built observations vs the live mjlab env.

For the SAME simulator state, compare the package's manual observation vector to
the env's ``observation_manager.compute_group(...)`` output — walker (47-d) and
jump arm (1175-d, incl. the 187-ray height-scan and the 5-frame history).
Asserts max|manual - mjlab| < 1e-4 for both.

Run (from the source repo):
  MUJOCO_GL=egl \
  PYTHONPATH=<repo>:<safety-stable-baselines> \
  ~/miniconda3/envs/mjlab/bin/python validation/validate_obs.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "/home/buzi/Desktop/RESEARCH/SAFE/DEVELOPMENT/go2_atomic_skills")

from go2_atomic_skills.obs import (MJLAB_JOINT_ORDER, DEFAULT_JOINT_POS,
                                   HistoryBuffer, ObsInputs, build_jump_frame,
                                   build_walker_obs)

TOL = 1e-4


def _np(t):
  return t.detach().cpu().numpy()


def _state(env):
  """Read the exact quantities the env's obs terms consumed this step."""
  robot = env.scene["robot"]
  d = robot.data
  quat = _np(d.root_link_quat_w[0]).astype(np.float32)     # w,x,y,z
  base_z = float(d.root_link_pos_w[0, 2])
  jpos = _np(d.joint_pos[0]).astype(np.float32)
  jvel = _np(d.joint_vel[0]).astype(np.float32)
  ang = _np(env.scene["robot/imu_ang_vel"].data[0]).astype(np.float32)
  cmd = _np(env.command_manager.get_command("twist")[0]).astype(np.float32)
  last_a = _np(env.action_manager.action[0]).astype(np.float32)
  step = int(env.episode_length_buf[0])
  return quat, base_z, jpos, jvel, ang, cmd, last_a, step


def check_joint_order(env):
  robot = env.scene["robot"]
  names = list(getattr(robot.data, "joint_names", None)
               or robot.joint_names)
  assert tuple(names) == tuple(MJLAB_JOINT_ORDER), \
    f"joint order mismatch:\n env={names}\n pkg={MJLAB_JOINT_ORDER}"
  dflt = _np(robot.data.default_joint_pos[0]).astype(np.float32)
  assert np.allclose(dflt, DEFAULT_JOINT_POS, atol=1e-6), \
    f"default_joint_pos mismatch: env={dflt} pkg={DEFAULT_JOINT_POS}"
  print(f"  joint order OK: {names}")
  print(f"  default_joint_pos OK: {dflt}")


def validate_walker():
  from robot_safety_sandbox.envs.velocity.go2 import unitree_go2_flat_env_cfg
  from mjlab.envs import ManagerBasedRlEnv
  print("\n=== WALKER (go2_walker_flat) ===")
  cfg = unitree_go2_flat_env_cfg(play=True)
  cfg.scene.num_envs = 1
  cfg.observations["actor"].enable_corruption = False
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  check_joint_order(env)
  obs_dict, _ = env.reset()
  worst = 0.0
  for i in range(8):
    grp = _np(env.observation_manager.compute_group("actor", update_history=False)[0])
    quat, base_z, jpos, jvel, ang, cmd, last_a, step = _state(env)
    inp = ObsInputs(joint_pos=jpos, joint_vel=jvel, base_ang_vel=ang,
                    base_z=base_z, base_quat=quat)
    manual = build_walker_obs(inp, cmd, last_a, step)
    d = np.abs(manual - grp).max()
    worst = max(worst, d)
    print(f"  step {i}: dim={manual.shape[0]} cmd={cmd.round(2)} step={step} "
          f"max|Δ|={d:.2e}")
    a = torch.as_tensor(np.random.uniform(-1, 1, (1, 12)), dtype=torch.float32,
                        device="cuda:0")
    env.step(a)
  env.close()
  print(f"  WALKER worst max|Δ| = {worst:.2e}  ->  {'PASS' if worst < TOL else 'FAIL'}")
  return worst


def validate_jump():
  from robot_safety_sandbox.envs.go2_gap.gap import unitree_go2_gap_reach_avoid_env_cfg
  from mjlab.envs import ManagerBasedRlEnv
  print("\n=== JUMP (go2_gap reach-avoid, proprioception 1175-d) ===")
  cfg = unitree_go2_gap_reach_avoid_env_cfg(play=True)
  cfg.scene.num_envs = 1
  cfg.observations["proprioception"].enable_corruption = False
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  check_joint_order(env)
  hist = HistoryBuffer()
  obs_dict, _ = env.reset()
  worst = 0.0
  scan_worst = 0.0
  for i in range(9):
    grp = _np(env.observation_manager.compute_group(
      "proprioception", update_history=False)[0])
    quat, base_z, jpos, jvel, ang, cmd, last_a, step = _state(env)
    # raw mdp.height_scan value (base_z - hit_z, miss -> max), UNSCALED.
    from mjlab.envs.mdp import height_scan as _hs
    scan_raw = _np(_hs(env, sensor_name="terrain_scan")[0]).astype(np.float32)
    inp = ObsInputs(joint_pos=jpos, joint_vel=jvel, base_ang_vel=ang,
                    base_z=base_z, base_quat=quat)
    frame = build_jump_frame(inp, cmd, last_a, step, scan_raw)
    hist.append(frame)
    manual = hist.flatten()
    d = np.abs(manual - grp).max()
    worst = max(worst, d)
    # gap signal sanity: how much does the forward scan vary this step
    scan_span = float(scan_raw.max() - scan_raw.min())
    scan_worst = max(scan_worst, scan_span)
    print(f"  step {i}: dim={manual.shape[0]} base_z={base_z:.3f} step={step} "
          f"scan[min,max]=[{scan_raw.min():.2f},{scan_raw.max():.2f}] max|Δ|={d:.2e}")
    a = torch.as_tensor(np.random.uniform(-1, 1, (1, 12)), dtype=torch.float32,
                        device="cuda:0")
    env.step(a)
  env.close()
  print(f"  JUMP worst max|Δ| = {worst:.2e}  ->  {'PASS' if worst < TOL else 'FAIL'}")
  print(f"  (height-scan value span seen: {scan_worst:.2f} m — gaps present)")
  return worst


def validate_action_parity():
  """End-to-end: package (obs build + extracted MLP + normalizer) vs the
  ORIGINAL SB3 / safety_sb3 policy, on live env states. Isolates nothing —
  this is the deployed runtime path compared to the real policy."""
  from mjlab.envs import ManagerBasedRlEnv
  from robot_safety_sandbox.envs.velocity.go2 import unitree_go2_flat_env_cfg
  from robot_safety_sandbox.envs.go2_gap.gap import unitree_go2_gap_reach_avoid_env_cfg
  from robot_safety_sandbox.eval.policies import load_nominal, load_twin
  import sys as _sys
  sys.path.insert(0, "/home/buzi/Desktop/RESEARCH/SAFE/DEVELOPMENT/go2_atomic_skills")
  from go2_atomic_skills.skills import WalkSkill, JumpSkill
  print("\n=== ACTION PARITY (package chain vs original policy) ===")

  # walker
  cfg = unitree_go2_flat_env_cfg(play=True); cfg.scene.num_envs = 1
  cfg.observations["actor"].enable_corruption = False
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  model, vn = load_nominal(
    "runs/go2_walker_flat/final_model.zip", "cpu", quiet=True)
  skill = WalkSkill("cpu")
  env.reset(); wworst = 0.0
  for i in range(6):
    quat, base_z, jpos, jvel, ang, cmd, last_a, step = _state(env)
    skill.step = step; skill.last_action = last_a.copy()
    inp = ObsInputs(joint_pos=jpos, joint_vel=jvel, base_ang_vel=ang,
                    base_z=base_z, base_quat=quat)
    obs = skill.raw_obs(inp, cmd)
    o = skill.norm(torch.as_tensor(obs).unsqueeze(0))
    a_pkg = skill.net(o).squeeze(0).detach().numpy()
    with torch.no_grad():
      a_ref = model.policy._predict(
        torch.as_tensor(vn.normalize_obs(obs).astype(np.float32)).unsqueeze(0),
        deterministic=True).squeeze(0).numpy()
    wworst = max(wworst, float(np.abs(a_pkg - a_ref).max()))
    env.step(torch.as_tensor(np.random.uniform(-1, 1, (1, 12)),
                             dtype=torch.float32, device="cuda:0"))
  env.close()
  print(f"  walker action max|Δa| = {wworst:.2e}")

  # jump
  cfg = unitree_go2_gap_reach_avoid_env_cfg(play=True); cfg.scene.num_envs = 1
  cfg.observations["proprioception"].enable_corruption = False
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  model, norm = load_twin("runs/gap_e040_resaved/ra_w30/final_model.zip", "cpu",
                          quiet=True)
  skill = JumpSkill("cpu", width="w30")
  from mjlab.envs.mdp import height_scan as _hs
  env.reset(); jworst = 0.0
  for i in range(7):
    quat, base_z, jpos, jvel, ang, cmd, last_a, step = _state(env)
    scan_raw = _np(_hs(env, sensor_name="terrain_scan")[0]).astype(np.float32)
    skill.step = step; skill.last_action = last_a.copy()
    inp = ObsInputs(joint_pos=jpos, joint_vel=jvel, base_ang_vel=ang,
                    base_z=base_z, base_quat=quat)
    frame = skill.raw_frame(inp, cmd, scan_raw)
    skill.history.append(frame)
    obs = skill.history.flatten()
    o = skill.norm(torch.as_tensor(obs).unsqueeze(0))
    a_pkg = skill.net(o).squeeze(0).detach().numpy()
    with torch.no_grad():
      a_ref = model.policy._predict(
        norm(torch.as_tensor(obs).unsqueeze(0)), deterministic=True
      ).squeeze(0).numpy()
    jworst = max(jworst, float(np.abs(a_pkg - a_ref).max()))
    env.step(torch.as_tensor(np.random.uniform(-1, 1, (1, 12)),
                             dtype=torch.float32, device="cuda:0"))
  env.close()
  print(f"  jump   action max|Δa| = {jworst:.2e}")
  return max(wworst, jworst)


if __name__ == "__main__":
  np.random.seed(0)
  torch.manual_seed(0)
  w = validate_walker()
  j = validate_jump()
  ap = validate_action_parity()
  print("\n================ SUMMARY ================")
  print(f"  walker 47-d   worst max|Δ| = {w:.2e}   {'PASS' if w < TOL else 'FAIL'}")
  print(f"  jump   1175-d worst max|Δ| = {j:.2e}   {'PASS' if j < TOL else 'FAIL'}")
  print(f"  action parity worst max|Δa| = {ap:.2e}   {'PASS' if ap < 1e-4 else 'FAIL'}")
  ok = (w < TOL) and (j < TOL) and (ap < 1e-4)
  print("  RESULT:", "ALL PASS" if ok else "FAIL")
  sys.exit(0 if ok else 1)

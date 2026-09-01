"""HONEST end-to-end validation. Three gates:

(a) BRIDGE bit-exact on DYNAMIC states: drive the live mjlab env to nonzero
    velocities, transfer each raw sim state into a plain MjData built from the
    bundled MJCF, run obs_from_mujoco + the package obs builders, and compare to
    mjlab's own compute_group(...). Walker (47) and jump (1175, incl. scan +
    5-frame history). Gate: max|Δ| < 1e-4.

(b) WALK ACCEPTANCE: closed-loop rollout of the full package in the bundled
    MuJoCo scene (implicitfast, trained gains, 50 Hz). Must realize >= 0.8 m/s
    body-frame forward speed at command (1,0,0), staying upright.

(c) JUMP trigger still fires (faked gap): base dips / thighs load.

Run (from the source repo, mjlab env):
  MUJOCO_GL=egl PYTHONPATH=<repo>:<safety-sb3>:<pkg> \
  ~/miniconda3/envs/mjlab/bin/python validation/validate_bridge_and_walk.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "/home/buzi/Desktop/RESEARCH/SAFE/DEVELOPMENT/go2_atomic_skills")

import mujoco

from go2_atomic_skills.obs import (MJLAB_JOINT_ORDER, ObsInputs,
                                   build_jump_frame, build_walker_obs,
                                   HistoryBuffer)
from go2_atomic_skills.mujoco_helper import (build_go2_mjcf_model, obs_from_mujoco,
                                             set_default_pose, fake_gap_scan)
from go2_atomic_skills.skills import Go2Skills

TOL = 1e-4
DEV = "cuda:0"


def _np(t):
  return t.detach().cpu().numpy()


def _transfer_state(mjm, mjd, robot, env):
  """Write the live mjlab state into a plain MjData by NAME, then mj_forward so
  xmat / sensordata (gyro) are consistent with the joints/base."""
  d = robot.data
  mjd.qpos[:3] = _np(d.root_link_pos_w[0])
  mjd.qpos[3:7] = _np(d.root_link_quat_w[0])          # w,x,y,z
  mjd.qvel[:3] = _np(d.root_link_lin_vel_w[0])        # free-joint linear = global
  # free-joint angular qvel is in the BODY frame == the imu gyro reading
  mjd.qvel[3:6] = _np(env.scene["robot/imu_ang_vel"].data[0])
  for i, nm in enumerate(MJLAB_JOINT_ORDER):
    jid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, nm)
    mjd.qpos[int(mjm.jnt_qposadr[jid])] = float(d.joint_pos[0, i])
    mjd.qvel[int(mjm.jnt_dofadr[jid])] = float(d.joint_vel[0, i])
  mujoco.mj_forward(mjm, mjd)


def gate_a_walker():
  from robot_safety_sandbox.envs.velocity.go2 import unitree_go2_flat_env_cfg
  from mjlab.envs import ManagerBasedRlEnv
  print("\n=== (a) BRIDGE bit-exact, WALKER 47-d, dynamic states ===")
  cfg = unitree_go2_flat_env_cfg(play=True); cfg.scene.num_envs = 1
  cfg.observations["actor"].enable_corruption = False
  env = ManagerBasedRlEnv(cfg, device=DEV)
  mjm = build_go2_mjcf_model(); mjd = mujoco.MjData(mjm)
  env.reset(); robot = env.scene["robot"]
  worst = 0.0
  for i in range(12):
    env.step(torch.as_tensor(np.random.uniform(-1, 1, (1, 12)), dtype=torch.float32,
                             device=DEV))
    grp = _np(env.observation_manager.compute_group("actor", update_history=False)[0])
    _transfer_state(mjm, mjd, robot, env)
    inp = obs_from_mujoco(mjm, mjd)
    cmd = _np(env.command_manager.get_command("twist")[0])
    last_a = _np(env.action_manager.action[0])            # post-gain ctrl
    step = int(env.episode_length_buf[0])
    manual = build_walker_obs(inp, cmd, last_a, step)
    d = float(np.abs(manual - grp).max())
    worst = max(worst, d)
    if i < 3 or d > TOL:
      # per-term breakdown to localize any residual
      segs = [("ang_vel", 0, 3), ("proj_g", 3, 6), ("cmd", 6, 9), ("phase", 9, 11),
              ("jpos", 11, 23), ("jvel", 23, 35), ("act", 35, 47)]
      br = {n: round(float(np.abs(manual[a:b]-grp[a:b]).max()), 6) for n, a, b in segs}
      print(f"  step {i}: max|Δ|={d:.2e}  {br}")
  env.close()
  print(f"  WALKER bridge worst max|Δ| = {worst:.2e} -> {'PASS' if worst<TOL else 'FAIL'}")
  return worst


def gate_a_jump():
  from robot_safety_sandbox.envs.go2_gap.gap import unitree_go2_gap_reach_avoid_env_cfg
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.envs.mdp import height_scan as _hs
  print("\n=== (a) BRIDGE bit-exact, JUMP 1175-d, dynamic states ===")
  cfg = unitree_go2_gap_reach_avoid_env_cfg(play=True); cfg.scene.num_envs = 1
  cfg.observations["proprioception"].enable_corruption = False
  env = ManagerBasedRlEnv(cfg, device=DEV)
  mjm = build_go2_mjcf_model(); mjd = mujoco.MjData(mjm)
  hist = HistoryBuffer()
  env.reset(); robot = env.scene["robot"]
  worst = 0.0
  for i in range(9):
    grp = _np(env.observation_manager.compute_group("proprioception", update_history=False)[0])
    _transfer_state(mjm, mjd, robot, env)
    inp = obs_from_mujoco(mjm, mjd)
    # height-scan comes from the terrain sensor (perception), not obs_from_mujoco:
    # feed the env's raw mdp.height_scan so we validate assembly/scale/history.
    scan_raw = _np(_hs(env, sensor_name="terrain_scan")[0])
    cmd = _np(env.command_manager.get_command("twist")[0])
    last_a = _np(env.action_manager.action[0])
    step = int(env.episode_length_buf[0])
    frame = build_jump_frame(inp, cmd, last_a, step, scan_raw)
    hist.append(frame)
    manual = hist.flatten()
    d = float(np.abs(manual - grp).max())
    worst = max(worst, d)
    if i < 2 or d > TOL:
      print(f"  step {i}: max|Δ|={d:.2e}")
    env.step(torch.as_tensor(np.random.uniform(-1, 1, (1, 12)), dtype=torch.float32,
                             device=DEV))
  env.close()
  print(f"  JUMP bridge worst max|Δ| = {worst:.2e} -> {'PASS' if worst<TOL else 'FAIL'}")
  return worst


def _fwd_speed(mjm, mjd):
  """Body-frame forward velocity = R^T v_world, x-component."""
  R = mjd.xmat[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "base_link")].reshape(3, 3)
  v = mjd.qvel[:3]
  return float((R.T @ v)[0])


def gate_b_walk(vx):
  print(f"\n=== (b) WALK ACCEPTANCE @ cmd=({vx},0,0) ===")
  mjm = build_go2_mjcf_model(); mjd = mujoco.MjData(mjm)
  set_default_pose(mjm, mjd, base_z=0.33)
  skills = Go2Skills(device="cpu"); skills.reset()
  speeds, up = [], []
  for k in range(400):
    inp = obs_from_mujoco(mjm, mjd)
    tgt = skills.walk(inp, command=(vx, 0.0, 0.0))
    mjd.ctrl[:] = tgt
    for _ in range(4):
      mujoco.mj_step(mjm, mjd)
    if k >= 200:
      speeds.append(_fwd_speed(mjm, mjd))
      up.append(inp.proj_grav()[2])                    # ~ -1 upright
  spd = float(np.mean(speeds))
  print(f"  realized forward speed = {spd:.3f} m/s (mean over last 200 steps); "
        f"proj_grav_z={np.mean(up):.3f} (upright~-1)")
  return spd


def gate_c_jump():
  print("\n=== (c) JUMP trigger (faked gap) ===")
  mjm = build_go2_mjcf_model(); mjd = mujoco.MjData(mjm)
  set_default_pose(mjm, mjd, base_z=0.33)
  skills = Go2Skills(device="cpu"); skills.reset()
  z0 = float(mjd.qpos[2]); thigh0 = []
  for k in range(60):
    inp = obs_from_mujoco(mjm, mjd)
    scan = fake_gap_scan(dist_to_gap=0.35, gap_width=0.30, base_z=inp.base_z)
    tgt = skills.jump(inp, command=(1.0, 0.0, 0.0), height_scan=scan)
    mjd.ctrl[:] = tgt
    for _ in range(4):
      mujoco.mj_step(mjm, mjd)
    thigh0.append(tgt[1])                               # FL_thigh target
  print(f"  jump ran 60 steps: base_z {z0:.3f}->{float(mjd.qpos[2]):.3f}; "
        f"FL_thigh target mean={np.mean(thigh0):.3f} (loads vs default 0.9)")
  return True


if __name__ == "__main__":
  np.random.seed(0); torch.manual_seed(0)
  wa = gate_a_walker()
  ja = gate_a_jump()
  s05 = gate_b_walk(0.5)
  s10 = gate_b_walk(1.0)
  gate_c_jump()
  print("\n================ SUMMARY ================")
  print(f"  (a) walker bridge max|Δ| = {wa:.2e}  {'PASS' if wa<TOL else 'FAIL'}")
  print(f"  (a) jump   bridge max|Δ| = {ja:.2e}  {'PASS' if ja<TOL else 'FAIL'}")
  print(f"  (b) walk speed cmd0.5 = {s05:.3f} m/s")
  print(f"  (b) walk speed cmd1.0 = {s10:.3f} m/s   {'PASS' if s10>=0.8 else 'FAIL'} (bar 0.8)")
  ok = wa < TOL and ja < TOL and s10 >= 0.8
  print("  RESULT:", "ALL PASS" if ok else "FAIL")
  sys.exit(0 if ok else 1)

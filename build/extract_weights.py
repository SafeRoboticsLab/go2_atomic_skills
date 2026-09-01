"""BUILD-TIME weight extraction (needs mjlab + safety_sb3 on PYTHONPATH).

Loads the two trained Go2 policies from the source repo, extracts their
DETERMINISTIC actor MLP (policy_net + action_net) into a plain
``torch.nn.Sequential`` state_dict, and writes the obs normalizers as plain
arrays. The delivered runtime then needs only torch + numpy (no SB3, no
safety_sb3).

Run:
  cd <source repo>
  MUJOCO_GL=egl \
  PYTHONPATH=<repo>:<safety-stable-baselines> \
  ~/miniconda3/envs/mjlab/bin/python <this package>/build/extract_weights.py

Every extracted MLP is verified against the original policy's deterministic
action on random obs (max abs diff < 1e-5).
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import torch
import torch.nn as nn

REPO = "/home/buzi/Desktop/RESEARCH/SAFE/DEVELOPMENT/safe_mjlab_zoo"
PKG = "/home/buzi/Desktop/RESEARCH/SAFE/DEVELOPMENT/go2_atomic_skills"
OUT = os.path.join(PKG, "go2_atomic_skills", "assets", "extracted")


def build_actor_sequential(policy) -> nn.Sequential:
  """Deterministic actor = mlp_extractor.policy_net (Linear/Tanh stack)
  followed by action_net (the Gaussian MEAN head). No output squashing:
  SB3's ``_predict(deterministic=True)`` returns this mean directly."""
  layers = list(policy.mlp_extractor.policy_net) + [policy.action_net]
  seq = nn.Sequential(*layers)
  seq.eval()
  return seq


def build_critic_sequential(policy) -> nn.Sequential:
  """State value V(s) = mlp_extractor.value_net (Linear/Tanh stack) followed by
  value_net (the scalar head). Matches SB3 ``policy.predict_values`` exactly for
  a FlattenExtractor (identity) features extractor, which the 1-D jump obs uses:
  ``predict_values`` is ``value_net(mlp_extractor.forward_critic(obs))`` and the
  features extractor is a no-op, so this Sequential reproduces it."""
  layers = list(policy.mlp_extractor.value_net) + [policy.value_net]
  seq = nn.Sequential(*layers)
  seq.eval()
  return seq


def verify(seq, policy, obs_dim, n=256, tag=""):
  seq.eval()
  rng = torch.Generator().manual_seed(0)
  obs = torch.randn(n, obs_dim, generator=rng) * 3.0  # spans the clipped range
  with torch.no_grad():
    a_ref = policy._predict(obs, deterministic=True)
    a_ext = seq(obs)
  d = (a_ref - a_ext).abs().max().item()
  print(f"  [{tag}] extracted-vs-original max|Δa| = {d:.2e}  (n={n})")
  assert d < 1e-5, f"{tag}: extracted MLP diverges ({d})"
  return d


def verify_critic(seq, policy, obs_dim, n=256, tag=""):
  seq.eval()
  rng = torch.Generator().manual_seed(0)
  obs = torch.randn(n, obs_dim, generator=rng) * 3.0  # spans the clipped range
  with torch.no_grad():
    v_ref = policy.predict_values(obs).squeeze(-1)
    v_ext = seq(obs).squeeze(-1)
  d = (v_ref - v_ext).abs().max().item()
  print(f"  [{tag}] extracted-vs-original max|ΔV| = {d:.2e}  (n={n})")
  assert d < 1e-5, f"{tag}: extracted critic diverges ({d})"
  return d


def extract_walker():
  from robot_safety_sandbox.eval.policies import load_nominal
  print("== WALKER (SB3 PPO, go2_walker_flat) ==")
  model, vn = load_nominal(
    os.path.join(REPO, "runs/go2_walker_flat/final_model.zip"), "cpu", quiet=True)
  policy = model.policy
  policy.set_training_mode(False)
  seq = build_actor_sequential(policy)
  verify(seq, policy, 47, tag="walker")
  torch.save(seq.state_dict(), os.path.join(OUT, "walker_actor.pt"))

  # VecNormalize obs statistics -> plain arrays. SB3 normalize_obs is
  #   clip((o - mean)/sqrt(var + epsilon), -clip_obs, clip_obs)
  rms = vn.obs_rms
  np.savez(
    os.path.join(OUT, "walker_norm.npz"),
    mean=rms.mean.astype(np.float32),
    var=rms.var.astype(np.float32),
    epsilon=np.float32(vn.epsilon),
    clip=np.float32(vn.clip_obs),
  )
  print(f"  walker_norm: mean{rms.mean.shape} clip={vn.clip_obs} eps={vn.epsilon}")


def extract_jump(width_tag, run_subdir):
  from robot_safety_sandbox.eval.policies import load_twin
  print(f"== JUMP {width_tag} (safety_sb3 ReachAvoidPPO1P, {run_subdir}) ==")
  zip_path = os.path.join(REPO, "runs/gap_e040_resaved", run_subdir, "final_model.zip")
  model, _norm = load_twin(zip_path, "cpu", quiet=True)
  policy = model.policy
  policy.set_training_mode(False)
  seq = build_actor_sequential(policy)
  verify(seq, policy, 1175, tag=f"jump_{width_tag}")
  torch.save(seq.state_dict(), os.path.join(OUT, f"jump_actor_{width_tag}.pt"))

  # The reach-avoid CRITIC V(s): what a value filter thresholds. Same 1175-d
  # normalized obs the actor consumes; safe iff V >= 0 (zoo sign convention).
  critic = build_critic_sequential(policy)
  verify_critic(critic, policy, 1175, tag=f"jump_critic_{width_tag}")
  torch.save(critic.state_dict(), os.path.join(OUT, f"jump_critic_{width_tag}.pt"))

  # tensornormalize.pt = {obs_mean(1175), obs_var(1175), count}. Applied as
  #   clip((o - mean)/sqrt(var + 1e-8), -10, 10)  (see policies.load_twin).
  st = torch.load(os.path.join(REPO, "runs/gap_e040_resaved", run_subdir,
                               "tensornormalize.pt"),
                  map_location="cpu", weights_only=True)
  torch.save({"obs_mean": st["obs_mean"].float(),
              "obs_var": st["obs_var"].float(),
              "epsilon": torch.tensor(1e-8),
              "clip": torch.tensor(10.0)},
             os.path.join(OUT, f"jump_norm_{width_tag}.pt"))
  print(f"  jump_norm_{width_tag}: mean{tuple(st['obs_mean'].shape)}")


if __name__ == "__main__":
  os.makedirs(OUT, exist_ok=True)
  extract_walker()
  extract_jump("w30", "ra_w30")
  extract_jump("w20", "ra_w20")
  extract_jump("w12", "ra_w12")
  print("\nDONE ->", OUT)

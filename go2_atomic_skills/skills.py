"""The atomic-skill API: WalkSkill, JumpSkill, and the Go2Skills facade.

Each skill OWNS its stateful bits (the gait-phase clock, the last raw action,
and — for the jump — the 5-frame history ring buffer). Call ``reset()`` at the
start of an episode.

Both policies output a 12-d action in [-1, 1], in mjlab JOINT order (the same
order as ``joint_pos`` / ``DEFAULT_JOINT_POS`` — verified against the env's
action term ``target_names``: action[i] drives joint i, NO regrouping). The
action is scaled by ``CTRL_GAIN`` (=3.0, the training/eval bridge's gain) and
the joint POSITION TARGET the PD controller tracks is
``default_joint_pos + 0.25 * (CTRL_GAIN * action)``. The skill methods return
that target; the post-gain ctrl (what the `actions` obs term holds) is kept on
``.last_action``.
"""

from __future__ import annotations

import numpy as np
import torch

from .nets import JumpNorm, WalkerNorm, load_actor, load_critic
from .obs import (ACTION_SCALE, CTRL_GAIN, DEFAULT_JOINT_POS, JUMP_FRAME_DIM,
                  SCAN_N, HistoryBuffer, ObsInputs, build_jump_frame,
                  build_walker_obs)


def _ctrl(policy_out: np.ndarray, clip: bool = True) -> np.ndarray:
  """Post-gain control vector: apply the bridge's ctrl_gain to the policy output.
  This is BOTH the `actions` obs term and the thing the joint-target formula
  scales.

  ``clip`` controls whether the raw action is clamped to [-1,1] BEFORE the gain.
  There are two action-processing paths in the SOURCE, and the DEPLOYMENT one is
  what this package reproduces:
    * The eval/deployment safety-FILTER path (``eval.policies.safety_modules``
      ``fallback_fn``) applies ``clamp(policy._predict, -1, 1)`` before the gain
      — so the arm engaged as a value filter feeds CLAMPED post-gain actions.
      This is what the validated composed result (E098) was measured under, and
      what the package's ``validate_value_filter`` faithfulness gate matches
      (worst |ΔV| back to ~1e-6 with clip=True). Hence the jump arm CLAMPS.
    * The raw training rollout (``base.py`` step_tensor) applies
      ``ctrl = action * ctrl_gain`` with no clamp, so a direct env.step with the
      raw policy mean stores an UNCLIPPED actions term. That path is reproduced
      by ``clip=False`` and kept only as an A/B knob — it is NOT the deployment
      target and desyncs from the validated filter (|ΔV|~0.2).
  The stock SB3 walker (numpy VecEnv path) clamps for the same reason SB3 does
  before stepping."""
  a = np.clip(policy_out, -1.0, 1.0) if clip else policy_out
  return (CTRL_GAIN * a).astype(np.float32)


def _target(ctrl: np.ndarray) -> np.ndarray:
  return (DEFAULT_JOINT_POS + ACTION_SCALE * ctrl).astype(np.float32)


class WalkSkill:
  """Atomic joystick walker (47-d single-frame obs, stock SB3 PPO)."""

  def __init__(self, device: str = "cpu"):
    self.device = device
    self.net = load_actor("walker_actor.pt", 47, device)
    self.norm = WalkerNorm(device)
    self.reset()

  def reset(self):
    self.step = 0
    self.last_action = np.zeros(12, dtype=np.float32)

  def raw_obs(self, inp: ObsInputs, command) -> np.ndarray:
    """The 47-d observation the policy consumes (pre-normalization)."""
    return build_walker_obs(inp, command, self.last_action, self.step)

  def act(self, inp: ObsInputs, command=(1.0, 0.0, 0.0)) -> np.ndarray:
    """One control step. Returns 12 joint position targets (mjlab order)."""
    obs = self.raw_obs(inp, command)
    o = self.norm(torch.as_tensor(obs, device=self.device).unsqueeze(0))
    with torch.no_grad():
      a = self.net(o).squeeze(0).cpu().numpy()
    ctrl = _ctrl(a)                      # walker: SB3-clipped post-gain (numpy VecEnv path)
    self.last_action = ctrl
    self.step += 1
    return _target(ctrl)


class JumpSkill:
  """Reach-avoid gap-jumping arm (1175-d = 235x5 history, ReachAvoidPPO1P).

  ``width`` picks the arm:
  'hando_w30' (DEFAULT) = the handover-range finetuned reach-avoid arm (gap
  0.30, finetuned on real walker-handover states — the recommended deployment
  arm); 'w30'/'w20'/'w12' = the from-scratch reverse-curriculum width ladder
  (0.30 / 0.20 / 0.12 m) kept as selectable alternates.

  ``with_critic=True`` also loads the reach-avoid state-value head V(s) (the
  same certificate the arm was trained under), enabling ``value_and_action`` and
  the :class:`Go2ValueFilter`. It is off by default so a bare arm needs only the
  actor."""

  def __init__(self, device: str = "cpu", width: str = "hando_w30",
               with_critic: bool = False, clip_action: bool = True):
    self.device = device
    self.width = width
    # The deployment safety-filter path clamps the arm action to [-1,1] before
    # the ctrl_gain (eval.policies.safety_modules), and that is the convention
    # the validated composed result was measured under — so clip_action=True by
    # default. clip_action=False reproduces the raw (unclamped) training-rollout
    # path and is an A/B knob only; see _ctrl.
    self.clip_action = bool(clip_action)
    self.net = load_actor(f"jump_actor_{width}.pt", 1175, device)
    self.norm = JumpNorm(width, device)
    self.critic = (load_critic(f"jump_critic_{width}.pt", 1175, device)
                   if with_critic else None)
    self.history = HistoryBuffer()
    self.reset()

  def reset(self):
    self.step = 0
    self.last_action = np.zeros(12, dtype=np.float32)
    self.history.reset()

  def raw_frame(self, inp: ObsInputs, command, height_scan: np.ndarray) -> np.ndarray:
    """The 235-d single frame (pre-history, pre-normalization)."""
    return build_jump_frame(inp, command, self.last_action, self.step, height_scan)

  def act(self, inp: ObsInputs, command=(1.0, 0.0, 0.0),
          height_scan: np.ndarray | None = None) -> np.ndarray:
    """One control step. ``height_scan`` is 187 raw ray heights (base_z - hit_z,
    miss -> 5.0); if None a flat-ground scan is synthesized (no gap -> the
    policy just walks/holds). Returns 12 joint position targets (mjlab order)."""
    if height_scan is None:
      height_scan = flat_ground_scan(inp.base_z)
    frame = self.raw_frame(inp, command, height_scan)
    self.history.append(frame)
    obs = self.history.flatten()
    o = self.norm(torch.as_tensor(obs, device=self.device).unsqueeze(0))
    with torch.no_grad():
      a = self.net(o).squeeze(0).cpu().numpy()
    ctrl = _ctrl(a, clip=self.clip_action)   # jump arm: deployment-filter clamp (clip_action, default True)
    self.last_action = ctrl
    self.step += 1
    return _target(ctrl)

  def value_and_action(self, inp: ObsInputs, command=(1.0, 0.0, 0.0),
                       height_scan: np.ndarray | None = None):
    """One control step that returns BOTH the certificate value and the action.

    Assembles the jump frame, pushes it onto the history, and normalizes ONCE,
    then evaluates the critic V(s) and the actor on the SAME normalized obs — so
    a single call advances the history/phase clock exactly once (the env's
    once-per-control-step semantics). Returns ``(V, ctrl)`` where ``V`` is the
    scalar reach-avoid value (safe iff >= 0) and ``ctrl`` is the post-gain
    control vector (12,). Requires the skill built with ``with_critic=True``."""
    if self.critic is None:
      raise RuntimeError(
        "JumpSkill.value_and_action needs the critic; build it with "
        "JumpSkill(..., with_critic=True) (or use Go2ValueFilter).")
    if height_scan is None:
      height_scan = flat_ground_scan(inp.base_z)
    frame = self.raw_frame(inp, command, height_scan)
    self.history.append(frame)
    obs = self.history.flatten()
    o = self.norm(torch.as_tensor(obs, device=self.device).unsqueeze(0))
    with torch.no_grad():
      a = self.net(o).squeeze(0).cpu().numpy()
      v = float(self.critic(o).squeeze(-1).cpu().item())
    ctrl = _ctrl(a, clip=self.clip_action)   # jump arm: deployment-filter clamp (clip_action, default True)
    self.last_action = ctrl
    self.step += 1
    return v, ctrl


def flat_ground_scan(base_z: float) -> np.ndarray:
  """Height scan over featureless flat ground: every ray hits z=0, so the raw
  mdp.height_scan value is base_z everywhere (187,)."""
  return np.full(SCAN_N, float(base_z), dtype=np.float32)


class Go2Skills:
  """Facade owning both skills.

      skills = Go2Skills(device="cpu")
      skills.reset()
      tgt = skills.walk(obs_inputs, command=(vx, vy, wz))
      tgt = skills.jump(obs_inputs, height_scan=None)
  """

  def __init__(self, device: str = "cpu", jump_width: str = "hando_w30"):
    self.device = device
    self.walk_skill = WalkSkill(device)
    self.jump_skill = JumpSkill(device, width=jump_width)

  def reset(self):
    self.walk_skill.reset()
    self.jump_skill.reset()

  def walk(self, obs_inputs: ObsInputs, command=(1.0, 0.0, 0.0)) -> np.ndarray:
    return self.walk_skill.act(obs_inputs, command)

  def jump(self, obs_inputs: ObsInputs, command=(1.0, 0.0, 0.0),
           height_scan: np.ndarray | None = None) -> np.ndarray:
    return self.jump_skill.act(obs_inputs, command, height_scan)


class Go2ValueFilter:
  """The reach-avoid arm deployed as a LEAST-RESTRICTIVE VALUE FILTER.

  Composes the walker (the nominal being filtered) with the jumping arm's
  certificate V(s) and its fallback action. The canonical switching rule, one
  memoryless comparison per step (Hsu, Hu & Fisac, "The Safety Filter"):

      apply the walker action   iff  V(s) >  eps
      else apply the jump action (hand authority to the certified fallback)

  So the arm engages when ``V(s) <= eps``. ``eps`` is the ONLY knob: 0.0 is the
  value-zero level (the certificate's own safe-set boundary); 0.25 is the tuned
  "shield earlier" value for the w30 arm; a larger eps engages the jump earlier
  / more conservatively. There is no latch, hysteresis, rest gate, or median
  smoothing here — that is a different, non-canonical variant.

  DEPLOYMENT CONTRACT (the switching-filter gotcha this class handles for you):
  the env computes BOTH obs groups from the current sim state, and the
  ``actions`` term in EACH group is the last *applied* (selected) control — not
  each policy's own previous output. So after every step both the walker's and
  the jump's ``last_action`` are set to the SELECTED post-gain control. Getting
  this wrong silently corrupts V; see ``validation/validate_value_filter.py``.
  """

  TRIGGERS = ("value", "distance")

  def __init__(self, device: str = "cpu", jump_width: str = "hando_w30",
               eps: float = 0.25, trigger: str = "value", D: float = 0.40,
               clip_action: bool = True):
    self.device = device
    self.eps = float(eps)
    if trigger not in self.TRIGGERS:
      raise ValueError(f"trigger must be one of {self.TRIGGERS}, got {trigger!r}")
    self.trigger = trigger
    self.D = float(D)
    self.walk_skill = WalkSkill(device)
    self.jump_skill = JumpSkill(device, width=jump_width, with_critic=True,
                               clip_action=clip_action)

  def reset(self):
    self.walk_skill.reset()
    self.jump_skill.reset()

  @staticmethod
  def dist_to_gap_from_scan(height_scan: np.ndarray, base_z: float,
                            drop_threshold: float = 0.7) -> float:
    """Estimate distance to the near edge of a forward gap from the raw 187-ray
    scan: the smallest FORWARD x-offset (high-ix column, y-center rows) whose
    drop (scan value - base_z) exceeds ``drop_threshold`` (the training gap
    definition). Returns ``inf`` if no forward drop is visible. Used by the
    distance-trigger when no explicit ``dist_to_gap`` is supplied."""
    from .obs import SCAN_NX, SCAN_NY, SCAN_X
    scan = np.asarray(height_scan, dtype=np.float32).reshape(SCAN_NY, SCAN_NX)
    cy = SCAN_NY // 2
    rows = scan[max(cy - 1, 0):cy + 2, :]          # 3 center y-rows, robust
    drop = rows - float(base_z)
    for ix in range(SCAN_NX):
      if SCAN_X[ix] > 0 and bool((drop[:, ix] > drop_threshold).any()):
        return float(SCAN_X[ix])
    return float("inf")

  def step(self, inp: ObsInputs, command=(1.0, 0.0, 0.0),
           height_scan: np.ndarray | None = None, eps: float | None = None,
           trigger: str | None = None, D: float | None = None,
           dist_to_gap: float | None = None):
    """One filtered control step. Returns ``(target, info)`` where ``target`` is
    the 12 joint position targets (mjlab order) and ``info`` carries
    ``{"value", "engaged", "eps", "trigger", "dist_to_gap"}``. ``height_scan``
    is the raw 187 ray heights (base_z - hit_z, miss -> 5.0); None synthesizes
    flat ground (no gap -> the arm holds, the walker passes through).

    Two engage modes:
      * ``trigger="value"``    — canonical least-restrictive filter: engage when
        ``V(s) <= eps``.
      * ``trigger="distance"`` — engage when distance to the gap's near edge is
        ``<= D`` (a decision LINE at the last brakeable point — E098's validated
        deployment mode). ``dist_to_gap`` may be passed explicitly (trivial with
        the fake-gap override, where it is known); otherwise it is estimated
        from the scan's first forward drop-off column.
    Both modes always evaluate V (reported in ``info``); only the engage test
    differs."""
    eps = self.eps if eps is None else float(eps)
    trigger = self.trigger if trigger is None else trigger
    D = self.D if D is None else float(D)
    # nominal (walker) post-gain ctrl; discard its target, keep the ctrl.
    self.walk_skill.act(inp, command)
    a_nom = self.walk_skill.last_action
    # certificate value + the jump's fallback ctrl, one history/phase advance.
    v, jump_ctrl = self.jump_skill.value_and_action(inp, command, height_scan)
    if trigger == "distance":
      if dist_to_gap is None:
        scan = height_scan if height_scan is not None else flat_ground_scan(inp.base_z)
        dist_to_gap = self.dist_to_gap_from_scan(scan, inp.base_z)
      engaged = dist_to_gap <= D
    else:
      dist_to_gap = float("nan") if dist_to_gap is None else dist_to_gap
      engaged = v <= eps
    applied_ctrl = jump_ctrl if engaged else a_nom
    # SHARED applied last_action into BOTH groups (see the deployment contract).
    self.walk_skill.last_action = applied_ctrl
    self.jump_skill.last_action = applied_ctrl
    target = _target(applied_ctrl)
    return target, {"value": float(v), "engaged": bool(engaged), "eps": eps,
                    "trigger": trigger, "dist_to_gap": float(dist_to_gap)}

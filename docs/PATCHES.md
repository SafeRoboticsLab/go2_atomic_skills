# Patches & migration guide — go2_atomic_skills

**If your project deployed `go2_atomic_skills` and the robot marches in place / crawls
at ~0.2 m/s / the jump never fires correctly — you have the broken build (v0.1.0).
Pull v0.2.0, or apply the fixes below.** The trained policies were always fine; every
bug here was in the *packaging* (how the action/obs are mapped outside mjlab).

Acceptance after fixing (verified end-to-end in a plain MuJoCo scene, closed loop
through `obs_from_mujoco`): **realized forward speed 0.40 m/s @ cmd 0.5, 0.88 m/s @
cmd 1.0** (0.88 is the policy's true ceiling — the same value it reaches in mjlab —
so command 1.0 realizes ~0.88, not exactly 1.0). Obs reconstruction is **bit-exact**
vs mjlab on dynamic (moving) states: walker 1.8e-7, jump 8.9e-8.

---

## THE load-bearing fix — `ctrl_gain = 3.0` (fixes the march-in-place / slow walk)

The go2 policies were trained through a bridge that **multiplies the policy action by
`ctrl_gain = 3.0` BEFORE** the env's `0.25` position-action scale, and the `actions`
(a.k.a. `last_action`) observation term holds the **post-gain** value. The broken
package used neither, so joint targets were ~4× too small → the robot barely moved.

Correct control law (per control step):

```
a      = policy(obs)                     # raw network output
a      = clip(a, -1, 1)                  # SB3 clips to the action space before stepping
ctrl   = 3.0 * a                         # <-- ctrl_gain = 3.0  (range now [-3, 3])
target = default_joint_pos + 0.25 * ctrl # == default + 0.75 * a   -> PD position target
```

and the **observation's `actions` term is `ctrl` (the post-gain value), NOT `a`.**

If you copied the old skill code, that is the change:
```python
# BROKEN (v0.1.0):
target = default_joint_pos + 0.25 * a
self.last_action = a
# FIXED (v0.2.0):
ctrl = 3.0 * np.clip(a, -1.0, 1.0)
target = default_joint_pos + 0.25 * ctrl
self.last_action = ctrl          # post-gain feeds next step's `actions` obs term
```

`ctrl_gain = 3.0` is the registry default for the go2 tasks (only car_goal=1.0 and
digit=12.0 override it). **Any redeploy of these Go2 checkpoints outside this package
must reapply the 3.0** or you get exactly the march-in-place failure.

---

## Every bug found, symptom → cause → fix

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Robot upright but **marches in place / ~0.2 m/s** regardless of command | Missing `ctrl_gain=3.0`; joint targets ~4× too small; `actions` obs used pre-gain value | Apply `ctrl = 3.0·clip(a)`; `target = default + 0.25·ctrl`; store `ctrl` as `last_action` (above) |
| 2 | Legs move in a **scrambled / non-forward gait** (only if you "fixed" #1 with a permutation) | **False lead:** the action is NOT reordered. `action[i]` drives joint `i` directly (env action term `target_ids = [0..11]`, leg-interleaved `[FL,FR,RL,RR]×[hip,thigh,calf]`). The grouped `gainprm=[20×8,40×4]` you may see is an internal MuJoCo *ctrl* layout, not the policy's action order | Do **not** permute the action. Apply it in the same leg-interleaved order as `DEFAULT_JOINT_POS` |
| 3 | `from go2_atomic_skills import fake_gap_scan` (or `obs_from_mujoco`, `raycast_height_scan`) `ImportError` | Helpers lived in `mujoco_helper` but weren't re-exported at top level | Re-exported in `__init__.py` (mujoco import stays lazy, so the numpy+torch core still imports without mujoco) |
| 4 | Suspected `obs_from_mujoco` wrong on moving states | **Not a bug** — the bridge is correct. Apparent mismatches were an artifact of an incorrect hand-rolled state-transfer test | Validation rewritten to transfer state faithfully (set `qvel[0:3]=lin_vel_w`, `qvel[3:6]=` body gyro, joints by dof address, then **`mj_forward`**) → all terms match to 0.0 |

**Deployment invariants (preserve these or the policy silently degrades):**
- Joint arrays (`joint_pos`, `joint_vel`, `DEFAULT_JOINT_POS`, targets) are in **mjlab
  leg-interleaved order**: `[FL,FR,RL,RR] × [hip,thigh,calf]`.
- `base_ang_vel` = body-frame IMU gyro; `projected_gravity` = `quat_apply_inverse(base_quat, [0,0,-1])`.
- `joint_pos` obs is **relative to the default pose** (`q - q_default`); `base_height` is absolute world-z.
- Gait `phase` is a **stateful clock** `[sin, cos]`, **zeroed when ‖command‖ < 0.1**;
  period **0.6 s (walker)**, **0.5 s (jump)**. The skill owns this — advance it once per control step.
- Jump obs is a **5-frame history** (oldest→newest) of the 235-d frame; `height_scan`
  is a **17×11=187** yaw-aligned down-grid, value `base_z − hit_z` (miss → 5.0), scaled ×0.2.
- Normalizers: walker `VecNormalize` (clip 10); jump `TensorNormalize` `(o-mean)/sqrt(var+1e-8)`.

**MuJoCo scene the policies expect (match in your sim):** PD position actuators
`kp/kd` = hip/thigh **20/1**, calf **40/2**; armature 0.01/0.01/0.02; **effort
(force) limits hip/thigh ±23.5 N·m, calf ±45 N·m** (`forcelimited`; the real Go2
torque limits the training sim enforces — omitting them lets the PD actuator apply
unbounded torque, so any package-sim validation is optimistic); integrator
**`implicitfast`**; timestep 0.005 with decimation 4 → **50 Hz** control; action scale
0.25 (× the ctrl_gain 3.0 above). `build_go2_mjcf_model()` sets all of this.

---

## How to migrate a project already on v0.1.0

1. **Preferred:** `git pull` / reinstall the package and `pip install -e .` again;
   confirm `go2_atomic_skills.__version__ == "0.2.0"`.
2. If you vendored/copied the skill code, apply fix #1 (the `ctrl_gain` control law) and
   make sure you did **not** add an action permutation (bug #2).
3. Re-run the acceptance check: a closed-loop walk at `command=(1.0,0,0)` must reach
   **≥0.8 m/s** and stay upright. `python validation/validate_bridge_and_walk.py`
   (needs `PYTHONPATH` to the mjlab repo for its bit-exact gate; the walk gate is
   standalone).
4. The jump needs a forward-facing gap in the height-scan + forward command — see
   `docs/JUMP_TRIGGER.md`. It is unchanged by these patches (same obs pipeline, so it
   inherits fix #1 too).

## Version history
- **0.4.3** — **landing latch ON by default** (`landing_latch=True, release="timed",
  release_delay_s=0.75`). This is the deployable setting from the package-sim landing
  study on the real 0.30 m gap with the scan-estimated trigger: memoryless 0.4.1 filter
  20-s survival 0.00 / backflip 0.24 → timed latch 0.87 / 0.04 (settle-gated release
  never fires on a hard landing; keep "timed"). The jump arm keeps its requested
  command through flight and touchdown, then the walker is released with a 0→cmd
  ramp over `release_ramp_s`. Later sandbox studies put the release-delay Pareto point
  at 0.5–0.75 s (both fine) and found that resuming the walker with a small forward
  command (~0.3 m/s) rather than zero avoids a slow backward creep on the far edge —
  keep `release_ramp_s` ≥ 0.5 s. `Go2ValueFilter(landing_latch=False)` restores the
  0.4.1 memoryless behaviour byte-for-byte (gate `validation/validate_landing_latch.py`
  now passes the flag explicitly). Also untracked the accidentally committed
  `__pycache__` / `*.egg-info` build files. Hardware prerequisites before a run:
  real joint gains equal to the sim contract (kp 20/40, kd 1/2), control latency
  ≤ 20 ms, low-friction front-foot caps, no payload, and start on a 0.20 m gap.
- **0.4.2** — **landing latch** on `Go2ValueFilter` (opt-in, `landing_latch=False`
  by default → 0.4.1 behavior byte-identical). The 0.4.0/0.4.1 filter is a
  MEMORYLESS per-step switch: with `trigger="distance"` it engages the jump only
  while `dist_to_gap ≤ D`, so once the robot is over/past the gap the forward
  scan shows no drop (`dist_to_gap → inf`) and authority snaps back to the WALKER
  *at or before touchdown* — a controller that never saw the landing pose, still
  commanded forward → the head-dive / anchored-front flip seen on hardware and in
  sim (jump-kept survival ~0.93 vs stay/walker handover 0.17/0.38 within 1 s;
  measured T028/E205). `trigger="value"` fails the same way (V > eps again
  mid-flight). The latch fixes the LANDING handover only (braking is unchanged):
  when the trigger first engages the jump it latches `engaged=True` and holds it
  through flight AND landing, with the jump kept at its REQUESTED command the whole
  time (never zeroed — a zeroed command is OOD for an arm trained under a constant
  command and causes a backward creep into the gap). Two release modes (`release=`):
  `"settle"` (default) releases after a settled-stand criterion holds CONTINUOUSLY
  for `settle_time_s` (precise; can fail to fire if the stand creeps); `"timed"`
  releases at a fixed `release_delay_s` (default 0.75) AFTER touchdown with no
  velocity gate (the sandbox composition's ≥0.90 20-s-survival mode) — touchdown =
  first foot contact after airborne (`foot_contacts` on the obs) else the
  base-height upturn (falling→not-falling) after descent. On release the WALKER's
  forward command is ramped 0→requested over `release_ramp_s` so it does not lurch
  off a fresh stand. Params (constructor + per-step `landing_latch`/`release`
  overrides): `settle_time_s=1.0`, `settle_speed=0.15` (base planar speed if the
  obs carries `base_lin_vel`), `settle_upright=0.8` (projected-gravity z),
  `settle_jointvel=4.0` (joint-velocity-norm fallback, since the shipped
  `ObsInputs` has neither foot contact nor base velocity), `release_delay_s=0.75`,
  `release_ramp_s=0.5`. Settled = all feet loaded (if `foot_contacts` present) else
  upright + slow. `info` gains `latched`, `settled_for_s`, `cmd_applied`,
  `touchdown`, `t_since_td`, `release`. The shared applied-last-action deployment
  contract is unchanged. Gate `validation/validate_landing_latch.py` (recorded
  approach→flight→settle sequence): OFF reproduces 0.4.1 **step-for-step** (worst
  |Δtarget| = 0, |ΔV| = 0, engage/dist mask 0/130 mismatches); ON (both release
  modes) stays engaged through the
  whole post-gap window the memoryless filter abandons and releases exactly at
  the state machine's predicted settled step. Everything else byte-identical
  (walker/jump nets, obs, `V`).
- **0.4.1** — **actuator effort limits** added to both MuJoCo actuator builders
  (`mujoco_helper.build_go2_mjcf_model` and the sweep's `add_actuators_and_arm`):
  `forcerange` ±23.5 N·m hip/thigh, ±45 N·m calf (`forcelimited` on), behind a
  single `_EFFORT_LIMITS` dict next to `_GAINS`. README:92-94 always documented
  these (the real Go2 limits the training sim enforces) but the code never set
  them, so package-sim validation was optimistic (unbounded torque). Everything
  else byte-identical. **Effect (before=unbounded vs after=limited):** the walk
  gate and the flat-floor jump trigger are *unchanged* — peak torque there stays
  well under the limits (walk @cmd1.0: HT 12.5, calf 22.4; trigger jump: HT 15.0,
  calf 32.5), so walk still realizes 0.882 m/s @cmd1.0 / 0.399 @cmd0.5 and the
  jump fires identically (0% steps clipped). Clipping only appears during a real
  gap crossing/landing, and only as brief impact transients: the unbounded sim
  overestimated peak torque by ~40 % (HT peaks 26–33 vs 23.5, calf 54–61 vs 45),
  but the clipped fraction is ≤0.3 % of substeps (HT) / ≤0.07 % (calf). Crossing
  rate on the shipped E098 `hando_w30` arm shifts ≤3 pp and only in already-
  marginal cells (fake-gap headline cells: gap0.20 D0.45 cross 1.00→1.00; gap0.30
  D0.40 cross 0.72→0.69); no documented gate breaks. The bit-exact bridge / net
  parity / value-filter gates are unaffected by construction (they compare obs/V
  on transferred states and never step the package's actuators).
- **0.4.0** — the **handover-range finetuned reach-avoid arm** becomes the DEFAULT
  jump (`hando_w30`): a gap-0.30 reach-avoid arm, 100M-step finetune on real
  walker-handover states (warm-started lineage), extracted bit-for-bit
  (`max|Δa| = 0`, `max|ΔV| = 0`) into `jump_{actor,critic,norm}_hando_w30.pt`.
  `JumpSkill` / `Go2Skills` / `Go2ValueFilter` default to it; the from-scratch
  reverse-curriculum width ladder (`w30`/`w20`/`w12`) stays selectable via
  `width=`/`jump_width=`. Bridge + net parity vs the source twin: single-frame
  obs `2.98e-8`, feed-forward actor/critic+normalizer `|Δa| 1.1e-6 / |ΔV| 1.8e-6`,
  closed-loop every term `<1e-7` except the env-randomized gait phase (which the
  package's clock owns). The `fake_gap_scan` override reproduces the arm's V and
  action from a real height-field gap with **100% V-sign agreement across the
  0.3-0.6 m deployment band** (gap 0.20 and 0.30). Adds a **`trigger`** option to
  `Go2ValueFilter`: `"value"` (canonical `V(s) <= eps`) or `"distance"` (engage
  at a decision LINE `dist_to_gap <= D`, default `D=0.40` — the arm's validated
  deployment mode; the gap distance is known under the fake-gap override and
  estimated from the scan's first forward drop-off otherwise). See
  `docs/VALUE_FILTER.md` for the recommended `eps`/`D` and the landing table.
  **Action-clamp clarification (supersedes the 0.4.0rc0 note):** the source has
  two action paths — the raw training rollout (`base.py` step_tensor,
  `ctrl = action*ctrl_gain`, NO clamp) and the eval/deployment safety-FILTER
  (`eval.policies.safety_modules` `fallback_fn`, which CLAMPS the arm action to
  [-1,1] before the gain). The validated composed result was measured under the
  clamped filter path, and `validate_value_filter` matches it (`|ΔV| 1.07e-6`)
  only with clamping — so the shipped jump arm **clamps** (`clip_action=True`,
  the default, same effective control law as ≤0.3.0). `clip_action=False`
  reproduces the unclamped rollout path and is kept only as an A/B knob.
- **0.3.0** — adds the reach-avoid **critic `V(s)`** and the **value filter**. The
  gap arm now ships its state-value head (`jump_critic_{w30,w20,w12}.pt`, extracted
  bit-for-bit: `max|ΔV| = 0` vs the source `predict_values`), a `load_critic` in
  `nets.py`, `JumpSkill.value_and_action`, and **`Go2ValueFilter`** — the arm
  deployed as a least-restrictive value filter on the walker (walk while `V(s) >
  eps`, jump when `V(s) ≤ eps`; `eps=0.25` default for w30). `Go2ValueFilter` owns
  the **shared applied-last-action** contract both obs groups need — the switching
  gotcha that silently drifts `V` if done by hand (see `docs/VALUE_FILTER.md`).
  Faithfulness gate `validation/validate_value_filter.py`: worst `|V_pkg -
  V_harness| = 1.1e-6`, engage mask exact (0/960) vs the source eval harness.
  No runtime deps added (still numpy+torch). Backward compatible: `JumpSkill`
  loads the critic only with `with_critic=True`.
- **0.2.0** — control-law fix (`ctrl_gain=3.0`, effective 0.75·a, post-gain `actions`
  obs); top-level re-export of the mujoco helpers; honest end-to-end bridge+walk
  validation; `build_go2_mjcf_model` uses `implicitfast`. **Walks at 0.88 m/s @ cmd 1.0.**
- **0.1.0** — initial package. **Broken:** missing `ctrl_gain`; robot marches in place
  (~0 m/s). Do not use.

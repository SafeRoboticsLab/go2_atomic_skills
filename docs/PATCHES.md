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

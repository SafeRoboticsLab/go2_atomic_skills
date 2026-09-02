# The gap arm as a least-restrictive value filter

The reach-avoid `JumpSkill` ships with its **critic** `V(s)` as well as its actor,
so it can be deployed not just as a controller but as a **safety filter** wrapping
another (nominal) policy — here the blind `WalkSkill`. `Go2ValueFilter` is that
composition, ready to drop in.

## What it is

A **least-restrictive value filter** (Hsu, Hu & Fisac, *"The Safety Filter: A
Unified View"*). One memoryless comparison per control step:

```
apply the WALKER action   iff  V(s) >  eps
else apply the JUMP action        (hand authority to the certified fallback)
```

Equivalently: **the jump engages when `V(s) <= eps`.** `V(s)` is the arm's own
reach-avoid state value — the certificate it was trained under. There is no
latch, hysteresis, rest gate, or median smoothing; the decision at step *t*
depends only on `V` at step *t*, so the arm holds authority for exactly as long
as the certificate says it must and returns it the instant it doesn't. That
minimality is what "least restrictive" means.

## What `V(s)` means

`V(s)` is the reach-avoid value from a `ReachAvoidPPO1P` critic (extracted
bit-for-bit — `max|ΔV| = 0` vs the source `predict_values`). **Sign convention:
safe iff `V >= 0`.** Loosely, `V` is high (positive, ~0.8 on flat standing
ground) when the arm is confident it can still reach the far side while avoiding
the gap from the current state, and drops toward / below 0 as the state
approaches the certificate's safe-set boundary (nearing the gap edge without a
committed crossing). It is a learned value, so read it as a calibrated margin,
not a physical distance.

## The `eps` knob (the only one)

`eps` shifts the engagement threshold along that margin:

| `eps` | meaning |
|------|---------|
| **0.0** | the value-zero level — the certificate's own safe-set boundary; engage exactly when the learned value says the state is no longer safe. |
| **0.25** | the tuned **"shield earlier"** value for the **w30** arm (from the source filter_traj sweep): hand to the arm a bit *before* the boundary, buying margin against a blind walker that would otherwise walk in too far. This is the package default. |
| **larger** | engages the jump **earlier / more conservatively** (a larger override set). |
| **negative** | spends margin — engages later, closer to (or past) the boundary. |

## Deployment contract — the switching-filter gotcha

Read this; it is the one non-obvious thing about deploying a switch. Each control
step the environment computes **both** observation groups (the walker's 47-d and
the jump's 1175-d) from the *current* sim state, and the **`actions` term inside
EACH group is the last *applied* (selected) control** — not each policy's own
previous output. So after every step you must set **both** the walker's and the
jump's `last_action` to the **selected** post-gain control:

```
applied = jump_ctrl  if  V(s) <= eps  else  walker_ctrl
walk_skill.last_action = applied          # BOTH groups carry the applied action
jump_skill.last_action = applied
```

If you instead let each skill keep its own previous output, the jump's 5-frame
history silently diverges from what the env would produce and `V(s)` drifts — the
classic switching-filter bug. **`Go2ValueFilter.step()` does this bookkeeping for
you**; the point is only that you must do it too if you compose the skills by
hand. The faithfulness gate `validation/validate_value_filter.py` is built to
catch exactly this: it compares the package's `V` to the source eval harness's on
the same live states and fails on any drift (measured worst `|ΔV| = 1.1e-06`,
engage mask exact over 960 comparisons).

## Minimal example

```python
import numpy as np
from go2_atomic_skills import Go2ValueFilter, ObsInputs, DEFAULT_JOINT_POS
from go2_atomic_skills.mujoco_helper import obs_from_mujoco, raycast_height_scan

filt = Go2ValueFilter(device="cpu", jump_width="w30", eps=0.25)
filt.reset()                          # clears both skills' phase/history/last-action

# one control step (50 Hz):
inp  = obs_from_mujoco(mj_model, mj_data)               # your robot state -> mjlab
scan = raycast_height_scan(mj_model, mj_data)           # 187 raw ray heights (or None)
target, info = filt.step(inp, command=(1.0, 0.0, 0.0), height_scan=scan)
#   target : 12 joint POSITION targets (mjlab joint order) -> mj_data.ctrl
#   info   : {"value": float V(s), "engaged": bool (jump active), "eps": float}

mj_data.ctrl[:] = target
# ... step 4 physics substeps, repeat ...
```

`height_scan` is the raw 187 ray heights (`base_z - hit_z`, miss -> 5.0), same as
`JumpSkill`; `None` synthesizes flat ground (no gap -> `V` stays high -> the
walker passes through). See [`docs/JUMP_TRIGGER.md`](JUMP_TRIGGER.md) for the scan
geometry and how to present a gap.

## Honest caveat — filter *quality* inherits the arm's momentum limit

The filter decides *when* to hand over faithfully; what happens *after* handover
is still the `JumpSkill`, and that arm **grazes at a slow approach**. Fed the
bundled walker's ~0.9 m/s, an engaged jump crosses gaps up to ~0.3 m but with a
low rear-leg/belly scramble (it was trained belly-allowed; a clean clear needs
~2.5 m/s launch the walker can't supply). So a "safe" crossing under this filter
means *did not fall in the gap*, not *cleared it cleanly*. Full data and the
reason there is no clean-landing variant in the package are in
[`docs/JUMP_TRIGGER.md`](JUMP_TRIGGER.md) §6.

---

## 0.4.0 — the default arm, the trigger option, and recommended settings

Since **0.4.0** the default jump is the **handover-range finetuned reach-avoid
arm** (`hando_w30`; gap 0.30, finetuned on real walker-handover states). It ships
with the same critic/value-filter machinery. Two things changed for deployment:
a **`trigger`** knob, and a per-arm recommended setting.

### `trigger="value"` vs `trigger="distance"`

```python
# E098 (default arm) — recommended deployment mode:
filt = Go2ValueFilter(device="cpu", trigger="distance", D=0.40)
# canonical value filter (calibrated for the w30 alternate):
filt = Go2ValueFilter(device="cpu", jump_width="w30", trigger="value", eps=0.25)
```

- **`value`** — the canonical least-restrictive filter: engage when `V(s) <= eps`.
- **`distance`** — engage at a decision LINE: when the distance to the gap's near
  edge is `<= D` (default `D=0.40`). This is the **validated deployment mode** for
  the handover arm (a single irrevocable jump/brake choice at the last brakeable
  point). The gap distance is known under the `fake_gap_scan` override (passed as
  `dist_to_gap`); with a real scan it is estimated from the first forward
  drop-off column (`Go2ValueFilter.dist_to_gap_from_scan`). Both modes always
  report `V` in `info`.

### Recommended settings (from the 0.4.0 sweep, N=32/cell)

| arm | recommended | why |
|---|---|---|
| **`hando_w30` (default, E098)** | **`trigger="distance", D=0.40`** | its `V` is more pessimistic than w30's, so the w30-tuned `eps` engages 0.6–1.4 m out and destabilizes; the distance line engages at the brakeable point and lands cleanly. |
| `w30` (E040b alternate) | `trigger="value", eps=0.25` (or `distance, D=0.30`) | the `eps=0.25` table below was tuned for this arm. |

**Do not** run the default arm with `trigger="value", eps=0.25`: that `eps` was
tuned for `w30` and, on the more-pessimistic handover critic, engages far too
early (the arm walks the whole approach under jump control and topples).

### Landing posture — the head-dive metric (FAKE override, flat ground, N=32)

Pitch sign: **`+` = nose up, `−` = nose down (dive)**. "head/trunk-first" =
fraction of crossings where a trunk/nose collision geom contacts the ground.

| arm · trigger | crossing | landing pitch mean / p90 | head/trunk-first |
|---|---|---|---|
| **`hando_w30` · distance D=0.40** | **0.72** | **+2.0° / +2.6°  (nose-up)** | **0.00** |
| `hando_w30` · distance D=0.30 | 0.56 | +2.4° / +3.2° | 0.00 |
| `w30` · value eps=0.25 | 0.84 | −8.2° / −5.6°  (nose-down) | 0.00 |
| `w30` · distance D=0.30 | 0.81 | −1.6° / −0.4° | 0.00 |

The handover arm lands **nose-up**; the old `w30` arm lands **nose-down** — the
head-dive symptom. Switching the default to `hando_w30` (and using the distance
trigger) is what removes the dive; it is a property of the arm, not the control
law.

### Real height-field gap at ~0.9 m/s (REAL mode, N=32)

Deployed as a live filter over the blind ~0.9 m/s walker, a **real** 0.30 m gap
is momentum-marginal for both arms — a clean clear needs the ~2.5 m/s launch the
walker cannot supply (see `docs/JUMP_TRIGGER.md` §6). The difference is the
FAILURE mode: the handover arm **refuses** (brakes/topples short; fall-INTO-gap
0.06–0.09) rather than committing and dropping in like `w30` (fall-in 0.69–0.84).
Refusing an uncrossable gap is the "avoid" half of reach-avoid working as
intended — the safer outcome.

### Faithfulness + parity (0.4.0)

- `validate_value_filter.py`: worst `|V_pkg − V_harness| = 1.07e-6`, engage mask
  exact (0/960) — matches the source value filter (the **clamped** deployment
  path; see `PATCHES.md` on the two action paths).
- `parity_hando_bridge.py`: single-frame bridge `2.98e-8`; feed-forward
  actor/critic `|Δa| 1.1e-6 / |ΔV| 1.8e-6`; closed-loop every term `<1e-7` except
  the env-randomized gait phase (the package owns its clock).
- `parity_hando_fakegap.py`: the `fake_gap_scan` override reproduces V and action
  from a real gap with **100% V-sign agreement across 0.3–0.6 m** (gap 0.20/0.30).

## Narrow-gap setpoint (0.18–0.20 m), default arm `hando_w30` — measured 2026-09-03

Package sim, fake-gap override, walker cmd 1.0, nominal gains, N=64 per cell. Crossing / survival /
landing-pitch p90 / head-first / peak non-foot force:

| D (distance trigger) | gap 0.20 | gap 0.18 |
|---|---|---|
| 0.40 | 0.98 / 0.98 / +3.4° / 0.00 / 1532 N | 0.25 / 0.25 / +1.6° / 0.12 / 1699 N |
| **0.45** | **1.00 / 1.00 / +3.4° / 0.00 / 0 N** | **0.97 / 1.00 / +2.6° / 0.00 / 0 N** |
| 0.50 | 1.00 / 1.00 / +2.7° / 0.00 / 0 N | 0.97 / 0.98 / +1.4° / 0.00 / 1340 N |

**Use `Go2ValueFilter(trigger="distance", D=0.45)` for 0.18–0.20 m gaps** — the only D window that
is ≥0.97 at both widths with feet-only landings. The value trigger does not cross with this arm at
narrow gaps (best 0.16 at ε=−0.5). The alternate `w30` arm with `eps=0.25` reaches only 0.84 / 0.75
at nominal gains and lands nose-down (mean −6.5°); its ~100% in sys1-sys2 relied on 2× PD gains.
Crossing mode here is a shallow low leap (base loft ~0.34–0.40 m), not a tall ballistic jump.

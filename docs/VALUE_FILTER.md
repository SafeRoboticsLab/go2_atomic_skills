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

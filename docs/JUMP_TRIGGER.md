# What makes the JumpSkill jump

The `JumpSkill` is **not** a button. It is a reach-avoid gap-jumping controller
whose behavior is driven by what the **height-scan** perceives: a gap ahead
while the robot is moving forward. It originates a jump when — and only when — it
both *sees a crossable gap in front of it* and *is approaching with forward
momentum/command*. This note documents, from the data, exactly what the policy
keys off and how to reproduce the trigger.

## 1. The scan geometry (what the policy sees of the terrain)

The last 187 numbers of each observation frame are a downward height-scan from a
yaw-aligned ray grid centered on the base:

- Grid: `size = (1.6 m, 1.0 m)`, `resolution = 0.1 m` → **17 cells in x
  (forward/back) × 11 cells in y (left/right) = 187 rays**.
- x offsets: `-0.8, -0.7, …, +0.8` m (positive x = **ahead**).
- y offsets: `-0.5, …, +0.5` m (y = 0 is the center row).
- `ray_alignment = "yaw"`: the grid is rotated to the base heading but kept
  **level** (not pitched with the body) — it always reads the ground plane
  around the robot's current heading.
- Rays point straight down; `max_distance = 5.0 m`.

**Ray ordering (row-major, y-major):** ray `k = iy*17 + ix`, with `iy` the y-row
(0…10, from y=-0.5 to +0.5) and `ix` the x-column (0…16, from x=-0.8 to +0.8).
So the **forward cells are the high-`ix` columns**, and the forward-center cells
are `iy = 5` (y = 0), `ix ≳ 10` (x ≳ +0.2 m).

**The value in each cell** is mjlab's `height_scan`:
`base_z − hit_z` (the base's height above whatever the ray hits), with a **miss**
(no hit within 5 m) mapping to `max_distance = 5.0`. It is then scaled by
`×0.2` (= 1/max_distance) before entering the network. On flat ground every cell
reads ≈ `base_z` (≈ 0.32 → 0.064 after scaling).

## 2. What a gap looks like in the scan

A gap is a trench in the floor. As the robot approaches, the **forward cells
(high `ix`) that fall over the void read a much larger distance**: instead of
`base_z − 0 ≈ 0.32 m`, they read `base_z − (−gap_depth)`, e.g. `≈ 1.3 m` for a
1 m-deep pit (validated live: the scan spans up to ~1.3 m over the gap edge), or
saturate to `5.0 m` (a miss) for a deep/wide gap. So the signature the policy
learned is:

> **near-side cells read flat-ground height; the forward cells suddenly jump to
> a large distance at the gap edge** — a drop-off in the direction of travel.

In the training reward a cell counted as "gap" when the drop exceeded
`drop_threshold = 0.7 m` (well above the ~0.3–0.4 m dips of rough/flat terrain),
so the policy's notion of a gap is a **clear, forward drop-off**, not terrain
roughness.

## 3. The conditions it was trained under

The gap policy (`ReachAvoidPPO1P`, the E040 arm; `ra_w30` = 0.30 m gap variant)
was trained on the reverse-curriculum gap env with:

- **Forward command ≈ 1.0 m/s** (the env pins `lin_vel_x = (1.0, 1.0)`, no
  lateral, no yaw). It expects to be *moving forward at ~1 m/s* — feeding it
  `command = 0` is out of distribution.
- **Belly-allowed posture:** body-height target 0.30 m and a very low
  `base_too_low` floor (0.05 m), so it may dip/extend the body to bridge.
- **Origination from a decision state:** the reverse curriculum starts committed
  (already airborne) and walks the start distribution back to a
  *standing/approaching* mixture near the near edge, so the policy learns to
  **originate** a jump from rest/approach, not only to finish one mid-air.

Empirical context (from the source project): as a deployed reach-avoid **filter**
at gap 0.30 m this arm reaches the far-side terminal on ≈ 45–100% of episodes
depending on the entry regime (100% in-curriculum with warm momentum; the cold
standstill edge is much harder, ~0.16–0.34 safe). The takeaway for reuse: it is
strongest when it enters *with* forward momentum and a clean gap in view.

## 4. How to fake a gap to invoke the jump

Use `fake_gap_scan(dist_to_gap, gap_width, base_z, gap_depth=1.0)`:

```python
from go2_atomic_skills.mujoco_helper import fake_gap_scan
scan = fake_gap_scan(dist_to_gap=0.35, gap_width=0.30, base_z=inp.base_z)
target = skills.jump(inp, command=(1.0, 0.0, 0.0), height_scan=scan)
```

It builds the 187-ray scan with **flat ground (`base_z`) everywhere except the
forward columns whose x-offset lies in `[dist_to_gap, dist_to_gap+gap_width]`,
which read the drop** (`base_z + gap_depth`, capped at 5.0). `dist_to_gap` must
be `< 0.8 m` to fall inside the scan's forward reach. With `full_width=True`
(default) the trench spans all 11 lateral rows — a real gap across the path.

**Be honest about what this is.** `fake_gap_scan` is *not* a pure trigger:
- It presents a gap **in the forward scan**. The policy still needs the rest of
  the observation to be consistent with "approaching a gap": a **forward
  command** (`command ≈ (1.0, 0, 0)`) and, ideally, actual forward velocity /
  the 5-frame history showing the gap getting nearer.
- The policy **originates a jump only when it judges the gap crossable** from
  that state. Present a gap it deems too wide (or with no forward momentum) and
  it will brake/hold instead — that refusal is the "avoid" half of reach-avoid,
  by design.
- On flat ground with **no** gap (`height_scan=None` → a flat scan), the jump
  policy just walks/holds; there is nothing to jump.

So the operational recipe to invoke origination is: **"drive a forward command,
feed a gap in the forward-center of the scan, and let the history show the
approach"** — then the reach-avoid policy decides whether to launch.

## 5. Practical knobs

- **Which width variant:** `Go2Skills(jump_width="w30" | "w20" | "w12")` selects
  the 0.30 / 0.20 / 0.12 m gap certificates. `w30` is the headline arm.
- **Distance to launch:** sweep `dist_to_gap` from ~0.6 → ~0.2 m across control
  steps (via the history) to emulate an approach; the policy commits as the edge
  enters its near-forward cells.
- **Momentum matters more than the scan alone.** If you can, enter with real
  forward velocity (base moving at ~1 m/s) rather than from a dead stop; the
  cold-standstill edge is the policy's hardest regime.

## 6. Jump *quality*: it grazes from a slow approach (expected, not a bug)

The most common surprise for a consuming project: the robot triggers the jump
and gets across, **but the rear legs / belly graze the floor instead of clearing
the gap cleanly.** This is inherent to the policy at low approach speed — it is
not a packaging bug (verified on the raw mjlab policy, no deployment involved).

**The jump's clearance is momentum-dependent.** Measured in the source env, the
base height *while over the gap* (standing height ≈ 0.32 m; a clean jump keeps
the base above that, a graze lets it sink toward the gap floor):

| approach at the edge | base-z over the gap | outcome |
|---|---|---|
| **committed / fast** (~2.8 m/s launch) | ~0.52 m | sails over cleanly |
| **grounded origination** (~1.5 m/s) | **~0.25 m** | crosses, but **rear grazes** |

Two compounding reasons:
1. **It was trained belly-allowed** (§3): body/leg contact was *not* penalized,
   so the policy never learned to avoid the scrape — scrambling across satisfies
   its objective. A graze is a success to it, not a failure.
2. **Clean clearance needs launch momentum** the walker can't supply. The bundled
   walker tops out at **~0.88 m/s** (see PATCHES.md), well below the ~2.5 m/s that
   produces a clean arc. Fed a ~0.9 m/s approach, the jump *will* graze.

**What to expect at ~0.9 m/s:** the robot crosses gaps up to ~0.3 m, but the
crossing is a low scramble with rear-leg/belly contact — the belly-allowed
behavior, working as trained. If your platform tolerates that contact, it's fine;
if you need a *clean* clear, you need a faster run-up (real velocity ≥ ~2 m/s at
the edge), which this package's walker cannot produce.

**Why there is no "clean-landing" (RAS lander) variant in the package.** A
certified airborne lander exists in the source project and *does* produce clean
landings — but it only takes over once the body is launched at **≥ 2.55 m/s**
(a geometric handover gate). From a grounded/slow origination the robot reaches
only ~1.5 m/s, the gate fires <half the time, and the base still sinks to ~0.25 m
— i.e. **at a slow approach the lander pipeline grazes identically to this arm**,
so shipping it would add two policies and a handover state machine for zero
improvement. It becomes worthwhile only if your robot can hit ~2.5 m/s at the
edge; if that ever becomes true for your platform, ask and it can be added.

## 7. The fake-gap override validated for the default (handover) arm (0.4.0)

Since **0.4.0** the default arm is the handover-range finetuned reach-avoid arm
(`hando_w30`). The `fake_gap_scan` override was re-checked against a REAL
height-field gap for this arm: for a physical state at a known distance, the
arm's `V(s)` and mean action on the synthetic scan were compared to the arm on a
real raycast of the same state (gap 0.20 and 0.30).

- **V-sign agreement is 100% across the 0.3–0.6 m deployment band** (both gap
  widths): the override drives the same jump/refuse decision the real gap would.
- Where the synthetic and real scans coincide (a grid cell not straddling the gap
  edge) `|ΔV| < 0.05` and `|Δa| < 0.05`; the scattered larger `|ΔV|` rows are the
  one grid column that flips at the gap boundary (an arm-independent quantization
  of `fake_gap_scan` vs the real geometry, `scan|Δ| = 1.0`).
- The only sign flip is at ~0.70 m — outside the deployment band, at the scan's
  forward-reach edge, where the real gap is barely visible and the arm correctly
  reads near-flat.

So `fake_gap_scan` remains a faithful way to invoke (or refuse) the handover arm.

**Distance trigger.** The handover arm's validated deployment mode is a decision
LINE, not a V-threshold: hand over when the gap is `<= D` m ahead (default
`D=0.40`). Under the override the distance is known and passed directly; with a
real scan it is estimated from the first forward drop-off column. See
`docs/VALUE_FILTER.md` — for `hando_w30`, `trigger="distance", D=0.40` gives the
cleanest crossings (nose-up landing, zero head/trunk-first contact) whereas the
w30-tuned value threshold engages this arm too early.

## 8. Which arm

`Go2Skills(jump_width="hando_w30" | "w30" | "w20" | "w12")` /
`JumpSkill(width=...)` select the arm. **`hando_w30`** (the handover-range
finetuned 0.30 m arm) is the **default** and the recommended deployment arm;
`w30`/`w20`/`w12` are the from-scratch reverse-curriculum width ladder (0.30 /
0.20 / 0.12 m), kept as selectable alternates.

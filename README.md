# go2_atomic_skills

> ⚠️ **Using v0.1.0 and the robot marches in place / crawls at ~0.2 m/s?** That build
> is broken (missing `ctrl_gain = 3.0`). Upgrade to **v0.2.0** and read
> **[`docs/PATCHES.md`](docs/PATCHES.md)** — it lists every bug + fix and how to migrate.
> Check your version: `python -c "import go2_atomic_skills as g; print(g.__version__)"`.

Two trained **Unitree Go2** policies packaged as reusable, MuJoCo-native
**atomic skills** for other projects. Drop it in, feed it your robot state, get
back joint targets — no `mjlab`, no `safety_sb3` at runtime (only `numpy`,
`torch`, and optionally `mujoco`).

| Skill | Class | Obs | Policy | What it does |
|------|-------|-----|--------|--------------|
| **Walk** | `WalkSkill` | 47-d single frame | stock SB3 PPO | joystick velocity walker (`vx, vy, ωz`) |
| **Jump** | `JumpSkill` | 1175-d (235 × 5 history) | `ReachAvoidPPO1P` | reach-avoid gap-jumping arm (actor **+ critic `V(s)`**); default = handover-range finetuned arm (0.4.0) |
| **Value filter** | `Go2ValueFilter` | both groups | walker + arm's `V(s)` | least-restrictive safety filter: walk until `V(s) ≤ eps`, then jump |

Both were extracted bit-for-bit from the source training runs; the reconstructed
observations match the live mjlab env to `< 1e-7` (including the `obs_from_mujoco`
bridge on dynamic states) and the extracted actor MLPs — and, since **0.3.0**, the
reach-avoid **critic `V(s)`** — reproduce the original policies to `0`. In the
bundled MuJoCo scene the walker realizes **0.88 m/s at command 1.0** (matching the
source mjlab walker), and the packaged value filter matches the source eval
harness's `V` to `< 1.1e-6` with an exact engage mask. See [Validation](#validation).

## Install

```bash
cd go2_atomic_skills
pip install -e .          # deps: numpy, torch   (+ pip install mujoco for the scene bridge)
```

## Quickstart

```python
import numpy as np
from go2_atomic_skills import Go2Skills, ObsInputs, DEFAULT_JOINT_POS

skills = Go2Skills(device="cpu")      # loads both policies + normalizers
skills.reset()                        # clears phase clock, last-action, jump history

# Fill from YOUR robot state (all joint arrays in mjlab joint order):
inp = ObsInputs(
    joint_pos    = DEFAULT_JOINT_POS.copy(),   # (12,) absolute joint angles
    joint_vel    = np.zeros(12),               # (12,)
    base_ang_vel = np.zeros(3),                # (3,) body-frame angular velocity
    base_z       = 0.32,                       # world z of the base (jump uses it)
    base_quat    = np.array([1,0,0,0]),        # (w,x,y,z); gives projected gravity
)

target = skills.walk(inp, command=(1.0, 0.0, 0.0))   # 12 joint POSITION targets
target = skills.jump(inp, command=(1.0, 0.0, 0.0),    # 12 joint POSITION targets
                     height_scan=None)                # None => flat scan (no gap => walks/holds)
```

Each call returns **12 joint position targets** (`default_joint_pos + 0.75 * a`
for policy output `a`, mjlab joint order). The post-gain `ctrl = 3.0*a` is on
`skills.walk_skill.last_action` / `skills.jump_skill.last_action`.

### From a MuJoCo scene

```python
from go2_atomic_skills.mujoco_helper import obs_from_mujoco, fake_gap_scan, raycast_height_scan

inp  = obs_from_mujoco(mj_model, mj_data)          # maps your joint order -> mjlab
scan = raycast_height_scan(mj_model, mj_data,      # real 17x11 down-scan (mj_ray)
                           geomgroup=terrain_mask)  # mask to terrain geoms only
target = skills.jump(inp, command=(1.0,0,0), height_scan=scan)
```

See [`example/demo.py`](example/demo.py) for a full closed-loop rollout (builds a
tiny Go2 scene from the bundled MJCF, walks, then jumps a faked gap).

## The Go2 setup (match these in your simulator)

- **Joint order** (mjlab): `FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
  RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf`. The policy's 12-d action
  is in this SAME joint order (`action[i]` drives joint `i` — verified against the
  env's action term `target_names`, no regrouping). `obs_from_mujoco` remaps from
  your model's names — pass `joint_names=[...]` if they differ.
- **Default joint pose:** `[-0.1, 0.9, -1.8,  0.1, 0.9, -1.8,  -0.1, 0.9, -1.8,
  0.1, 0.9, -1.8]`; base spawn `z ≈ 0.32`.
- **Action → control (important):** the policy outputs `a ∈ [-1,1]¹²`. The
  training/eval bridge applied a **`ctrl_gain = 3.0`** before the env, and the
  env's `0.25` position scale is on top of that, so the joint position target is
  `default_joint_pos + 0.25 * (3.0 * a) = default_joint_pos + 0.75 * a`. The skill
  methods already return this target; the post-gain `ctrl = 3.0*a` (what the
  `actions` obs term holds, and what `.last_action` stores) is handled internally.
  If you drive the extracted MLP yourself, **do not forget the 3.0** — omitting it
  makes the robot march in place (~0 m/s).
- **Actuator PD gains** (position control): hip & thigh `stiffness=20, damping=1,
  effort_limit=23.5, armature=0.01`; calf `stiffness=40, damping=2,
  effort_limit=45, armature=0.02`. Use the `implicitfast` integrator to match
  mjlab. `mujoco_helper.build_go2_mjcf_model()` sets all of this up for you.
- **Control rate:** 50 Hz (sim `timestep=0.005`, `decimation=4`). Call a skill
  once per control step; run 4 physics substeps between calls.
- **Realized walking speed** (bundled scene, full closed loop through
  `obs_from_mujoco`): **0.88 m/s at command (1.0, 0, 0)**, 0.40 m/s at (0.5,0,0).

## Observation specs (what the package reconstructs)

Both reproduced bit-for-bit from the mjlab observation manager.

**Walker — `actor` group, 47-d single frame, concat order:**
`base_ang_vel(3)`, `projected_gravity(3)`, `command(3)`, `phase(2)`,
`joint_pos(12, rel to default)`, `joint_vel(12)`, `actions(12, prev raw)`.
- `phase = [sin(2πp), cos(2πp)]`, `p = (step·dt mod T)/T`, **T = 0.6 s**; zeroed
  when `‖command‖ < 0.1`.

**Jump — `proprioception` group, 1175-d = 235-d frame × 5-frame history
(term-major; per-term scale applied before stacking; then a full-vector
TensorNormalize).** Single-frame (235-d) order:
`base_ang_vel(3)`, `projected_gravity(3)`, `command(3)`, `phase(2, T=0.5 s)`,
`base_height(1, world z)`, `joint_pos(12)`, `joint_vel(12)`, `actions(12)`,
`height_scan(187, ×0.2)`.
- History is chronological oldest→newest; the 1175 vector is **term-major**: for
  each of the 9 terms, its 5 frames flattened, in the order above. On reset the
  first frame backfills all 5 slots (mjlab CircularBuffer semantics).
- `height_scan`: 17×11 yaw-aligned down-ray grid, value `base_z − hit_z`
  (miss → 5.0), scaled ×0.2. See [`docs/JUMP_TRIGGER.md`](docs/JUMP_TRIGGER.md).

The package OWNS the stateful bits: the gait-phase clock, the previous raw
action, and (jump) the 5-frame history ring buffer. `reset()` clears all three.

## Invoking the jump — read this

The jump policy is perception-driven: it jumps when it **sees a gap ahead in the
height-scan while moving forward**. It is not a pure button. Full detail —
scan geometry, what a gap looks like, the trained approach conditions, and honest
caveats — in **[`docs/JUMP_TRIGGER.md`](docs/JUMP_TRIGGER.md)**. In short:

```python
from go2_atomic_skills.mujoco_helper import fake_gap_scan
scan = fake_gap_scan(dist_to_gap=0.35, gap_width=0.30, base_z=inp.base_z)
target = skills.jump(inp, command=(1.0, 0.0, 0.0), height_scan=scan)
```

## Value filter — the arm as a safety filter

Since **0.3.0** the reach-avoid arm ships its **critic `V(s)`** too, so it can wrap
the blind walker as a **least-restrictive value filter**: apply the walker while
`V(s) > eps`, hand over to the jump when `V(s) ≤ eps`. Since **0.4.0** the default arm is the **handover-range finetuned** reach-avoid
arm, whose recommended deployment mode is a decision LINE — `trigger="distance",
D=0.40` (the `eps` value-threshold below is calibrated for the `w30` alternate).
`eps` is the value knob
(`0.0` = the certificate's value-zero boundary; `0.25` = the tuned "shield
earlier" default for w30; larger = engage sooner). `Go2ValueFilter` owns the
whole composition — including the **shared applied-last-action** bookkeeping both
obs groups need (get that wrong by hand and `V` silently drifts). Full detail, the
`eps` table, the deployment contract, and honest caveats:
**[`docs/VALUE_FILTER.md`](docs/VALUE_FILTER.md)**.

```python
from go2_atomic_skills import Go2ValueFilter
from go2_atomic_skills.mujoco_helper import obs_from_mujoco, raycast_height_scan

filt = Go2ValueFilter(device="cpu", trigger="distance", D=0.40)  # default arm = handover RA; 0.4.3: landing latch ON (release="timed")
filt.reset()

inp  = obs_from_mujoco(mj_model, mj_data)
scan = raycast_height_scan(mj_model, mj_data)     # 187 raw ray heights (or None -> flat)
target, info = filt.step(inp, command=(1.0, 0.0, 0.0), height_scan=scan)
#  target -> mj_data.ctrl ;  info = {"value": V, "engaged": bool, "eps": eps}
```

## Limitations (honest)

- **The jump needs a terrain height-scan** (perception). With `height_scan=None`
  (flat, no gap) the jump policy just walks/holds — there's nothing to jump. Use
  `fake_gap_scan` or a real `raycast_height_scan` to present a gap.
- **`fake_gap_scan` is not a guaranteed trigger** — it presents a gap; the
  reach-avoid policy still decides whether to launch, and refuses gaps it judges
  uncrossable or entered from a dead stop (that refusal is the "avoid" half).
- **`raycast_height_scan` is best effort.** `mujoco.mj_ray` can exclude only one
  body, so downward rays may hit the robot's own legs; pass a `geomgroup` mask
  isolating your terrain geoms to match mjlab's terrain-only scan.
- **Gains/rate must match.** The policies assume the PD gains and 50 Hz control
  above. Different gains → different closed-loop behavior.
- The walker is blind (no scan); it knows nothing about gaps.
- **The jump grazes from a slow approach — expected, not a bug.** At the walker's
  ~0.9 m/s the robot crosses but the rear legs/belly scrape (it was trained
  belly-allowed; a clean clear needs ~2.5 m/s launch the walker can't supply).
  Full explanation + data in [`docs/JUMP_TRIGGER.md`](docs/JUMP_TRIGGER.md) §6.

## Layout

```
go2_atomic_skills/
  go2_atomic_skills/
    __init__.py        skills.py     # API: Go2Skills, WalkSkill, JumpSkill, Go2ValueFilter
    obs.py             nets.py       # obs reconstruction; portable MLPs (actor + critic) + normalizers
    mujoco_helper.py                 # obs_from_mujoco, raycast_height_scan,
                                     #   fake_gap_scan, build_go2_mjcf_model
    assets/
      extracted/                     # RUNTIME: actor + critic MLP state_dicts + normalizers
      raw/                           # provenance: original SB3/safety_sb3 zips + normalizers
      go2_mjcf/                      # Go2 MJCF + meshes (for the demo / raycast)
  build/extract_weights.py           # BUILD-TIME: extract actor + critic MLPs from the source runs
  validation/
    validate_bridge_and_walk.py      # honest gate: bridge bit-exact + walk speed
    validate_obs.py                  # obs assembly + action parity
    validate_value_filter.py         # honest gate: V(s) + engage mask vs the source harness
  example/demo.py  example/minimal.py
  docs/JUMP_TRIGGER.md  docs/VALUE_FILTER.md
```

## Validation

Two scripts, both run from the source repo (`safe_mjlab_zoo`) with the mjlab
conda env — mjlab + safety_sb3 are needed only for the comparison, never for the
shipped package.

**`validation/validate_bridge_and_walk.py`** — the honest end-to-end gate
(`validation/validation_output.txt`):

```
(a) obs_from_mujoco BRIDGE bit-exact on DYNAMIC states (state transferred into a
    plain MjData built from the bundled MJCF, then compared to mjlab):
      walker  47-d   worst max|Δ| = 1.79e-07   PASS
      jump   1175-d  worst max|Δ| = 8.94e-08   PASS   (incl. 187-ray scan + history)
(b) WALK acceptance (full closed loop, bundled scene, implicitfast):
      cmd (0.5,0,0) -> 0.399 m/s
      cmd (1.0,0,0) -> 0.882 m/s   PASS (bar 0.8), upright (proj_grav_z=-1.0)
(c) JUMP trigger fires on a faked gap (base dips, thighs load).
```

**`validation/validate_obs.py`** — obs ASSEMBLY vs the env + action parity vs the
original policies (`validation/validation_output_assembly.txt`):

```
walker 47-d   worst max|Δ|  = 8.94e-08   PASS
jump  1175-d  worst max|Δ|  = 8.94e-08   PASS
action parity worst max|Δa| = 5.96e-08   PASS
```

**`validation/validate_value_filter.py`** — the value filter (critic + engage
decision) vs the source eval harness's `build_filter("value", …)` on the RA w30
arm, per-env in lockstep over live states (`validation/validation_output_value_filter.txt`):

```
critic extraction  max|ΔV| = 0.0        (vs predict_values, at build time)
V(s)  worst |V_pkg - V_harness| = 1.07e-06   PASS (tol 1e-4)
engage mask (V ≤ eps=0.25)  0 / 960 mismatches   PASS
```

"""Unit test for the 0.4.2 landing latch on ``Go2ValueFilter``.

Two claims, checked on a recorded approach -> flight -> settle input sequence
(``validation/fixtures_latch_seq.npz``, generated from the 0.4.1 filter):

  (1) LATCH OFF reproduces 0.4.1 STEP-FOR-STEP. The recorded (target, value,
      engaged, dist_to_gap) from the 0.4.1 Go2ValueFilter must be reproduced
      bit-for-bit by the 0.4.2 filter with ``landing_latch=False``. This is the
      backward-compatibility gate: default-OFF is byte-identical.

  (2) LATCH ON stays engaged PAST THE GAP until settled. On the same sequence
      the 0.4.1/OFF filter disengages the jump the moment the gap leaves the
      forward scan (``dist_to_gap -> inf``) — i.e. it hands authority to the
      walker at touchdown. With ``landing_latch=True`` the filter must instead
      hold ``engaged=True`` through that whole window and release only after the
      settled-stand criterion has held continuously for ``settle_time_s`` — at
      exactly the step the state machine predicts.

Pure numpy+torch (no mujoco). Run:
    CUDA_VISIBLE_DEVICES="" python validation/validate_landing_latch.py
Exit code 0 iff both claims hold.
"""
from __future__ import annotations
import os, sys
import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
from go2_atomic_skills.obs import CONTROL_DT, ObsInputs
from go2_atomic_skills.mujoco_helper import fake_gap_scan
from go2_atomic_skills.skills import Go2ValueFilter

FIX = os.path.join(PKG, "validation", "fixtures_latch_seq.npz")


def _inputs(fx):
  N = fx["joint_pos"].shape[0]
  seq = []
  for k in range(N):
    seq.append(ObsInputs(
      joint_pos=fx["joint_pos"][k], joint_vel=fx["joint_vel"][k],
      base_ang_vel=fx["base_ang_vel"][k], base_z=float(fx["base_z"][k]),
      projected_gravity=fx["projected_gravity"][k]))
  return seq


def _scan(fx, k):
  return fake_gap_scan(float(fx["dist"][k]), float(fx["gap"]), float(fx["base_z"][k]))


def test_off_byte_identical(fx, seq):
  cmd = tuple(float(c) for c in fx["command"])
  D = float(fx["D"]); width = str(fx["width"])
  f = Go2ValueFilter(device="cpu", jump_width=width, trigger="distance", D=D,
                     landing_latch=False)
  f.reset()
  worst_t = 0.0; worst_v = 0.0; eng_mismatch = 0; dg_mismatch = 0
  for k, inp in enumerate(seq):
    t, info = f.step(inp, command=cmd, height_scan=_scan(fx, k), dist_to_gap=None)
    worst_t = max(worst_t, float(np.max(np.abs(t - fx["ref_target"][k]))))
    worst_v = max(worst_v, abs(float(info["value"]) - float(fx["ref_value"][k])))
    eng_mismatch += int(bool(info["engaged"]) != bool(fx["ref_engaged"][k]))
    ref_dg = float(fx["ref_dist_to_gap"][k]); got_dg = float(info["dist_to_gap"])
    if not (np.isnan(ref_dg) and np.isnan(got_dg)) and not np.isclose(ref_dg, got_dg, atol=0, rtol=0):
      dg_mismatch += 1
    # OFF must not carry any latch state.
    assert info["latched"] is False and info["settled_for_s"] == 0.0
  ok = (worst_t == 0.0 and worst_v == 0.0 and eng_mismatch == 0 and dg_mismatch == 0)
  print(f"[OFF byte-identity] worst|Δtarget|={worst_t:.2e} worst|ΔV|={worst_v:.2e} "
        f"engage_mismatch={eng_mismatch}/{len(seq)} dist_mismatch={dg_mismatch}/{len(seq)} "
        f"-> {'PASS' if ok else 'FAIL'}")
  return ok


def test_on_latches_past_gap(fx, seq):
  cmd = tuple(float(c) for c in fx["command"])
  D = float(fx["D"]); width = str(fx["width"])
  # Test-chosen latch params so a release occurs WITHIN this recorded sequence
  # (defaults are more conservative). The expected release step is derived from
  # the same settled criterion, so the assertion tracks the state machine.
  settle_time_s = 0.30           # 15 control steps @ 50 Hz
  settle_jointvel = 3.0
  settle_upright = 0.8
  f = Go2ValueFilter(device="cpu", jump_width=width, trigger="distance", D=D,
                     landing_latch=True, release="settle", settle_time_s=settle_time_s,
                     settle_jointvel=settle_jointvel, settle_upright=settle_upright,
                     release_ramp_s=0.5)
  f.reset()
  on_engaged = []
  for k, inp in enumerate(seq):
    _t, info = f.step(inp, command=cmd, height_scan=_scan(fx, k), dist_to_gap=None)
    on_engaged.append(bool(info["engaged"]))
  on_engaged = np.array(on_engaged, dtype=bool)
  off_engaged = fx["ref_engaged"].astype(bool)

  # --- derive the EXPECTED latch trace from the settled criterion ---
  n_settle = int(round(settle_time_s / CONTROL_DT))
  latch_on_step = int(np.argmax(off_engaged))          # first raw engage
  assert off_engaged[latch_on_step], "fixture never engages the jump"
  expect = np.zeros(len(seq), dtype=bool)
  latched = False; settled = 0; released_at = None
  for k, inp in enumerate(seq):
    if off_engaged[k] and not latched:                 # raw trigger latches it
      latched = True; settled = 0
    if latched:
      pg_up = float(inp.proj_grav()[2]) <= -settle_upright
      slow = float(np.linalg.norm(np.asarray(inp.joint_vel, np.float32))) <= settle_jointvel
      settled = settled + 1 if (pg_up and slow) else 0
      if settled >= n_settle:
        latched = False
        if released_at is None:
          released_at = k
    expect[k] = latched
  assert released_at is not None, "expected a release within the sequence"

  # (a) exact match to the predicted latch trace
  trace_ok = bool(np.array_equal(on_engaged, expect))
  # (b) the headline property: in the window after OFF has disengaged but before
  #     the ON release, ON is engaged where OFF is NOT (jump held past the gap).
  off_disengage = int(np.argmax(~off_engaged[latch_on_step:]) + latch_on_step)
  window = slice(off_disengage, released_at)
  held_past_gap = bool(np.all(on_engaged[window]) and not np.any(off_engaged[window])
                       and (released_at - off_disengage) > 0)
  ok = trace_ok and held_past_gap
  print(f"[ON latch] latch@{latch_on_step} OFF_disengage@{off_disengage} "
        f"ON_release@{released_at} (held {released_at - off_disengage} steps past "
        f"OFF-handover) trace_exact={trace_ok} held_past_gap={held_past_gap} "
        f"-> {'PASS' if ok else 'FAIL'}")
  return ok


def test_on_timed_release(fx, seq):
  """release='timed': latch holds past the gap and releases release_delay_s after
  touchdown (detected from the base-height upturn after descent), tracking the
  state machine exactly."""
  cmd = tuple(float(c) for c in fx["command"])
  D = float(fx["D"]); width = str(fx["width"])
  release_delay_s = 0.30                            # 15 control steps @ 50 Hz
  f = Go2ValueFilter(device="cpu", jump_width=width, trigger="distance", D=D,
                     landing_latch=True, release="timed",
                     release_delay_s=release_delay_s, release_ramp_s=0.5)
  f.reset()
  on_engaged = []; td_reported = None
  for k, inp in enumerate(seq):
    _t, info = f.step(inp, command=cmd, height_scan=_scan(fx, k), dist_to_gap=None)
    on_engaged.append(bool(info["engaged"]))
    if td_reported is None and info["touchdown"]:
      td_reported = k
  on_engaged = np.array(on_engaged, dtype=bool)
  off_engaged = fx["ref_engaged"].astype(bool)

  # expected trace: mirror the filter's accounting exactly (t_since_td increments
  # on the touchdown frame itself, so release fires at delay - one control step).
  bz = fx["base_z"].astype(np.float64)
  latch_on_step = int(np.argmax(off_engaged))
  latched = False; descended = False; prev_vz = None; prev_bz = None
  td = False; t_since_td = 0.0; td_step = None; release_step = None
  expect = np.zeros(len(seq), dtype=bool)
  for k in range(len(seq)):
    if off_engaged[k] and not latched and not (td or release_step is not None):
      latched = True
    if latched:
      if not td:                                   # touchdown detection (base-z)
        if prev_bz is not None:
          vz = (bz[k] - prev_bz) / CONTROL_DT
          if vz < -0.15: descended = True
          if descended and prev_vz is not None and prev_vz < 0.0 and vz >= 0.0 and bz[k] > 0.18:
            td = True; td_step = k
          prev_vz = vz
        prev_bz = bz[k]
      if td:
        t_since_td += CONTROL_DT
      if td and t_since_td >= release_delay_s:
        latched = False; release_step = k
    expect[k] = latched
  assert td_step is not None and release_step is not None, "expected touchdown+release in seq"
  trace_ok = bool(np.array_equal(on_engaged, expect))
  td_ok = (td_reported == td_step)
  # held past the gap: engaged where OFF is not, from OFF-disengage to release
  off_disengage = int(np.argmax(~off_engaged[latch_on_step:]) + latch_on_step)
  window = slice(off_disengage, min(release_step, len(seq)))
  held = bool(np.all(on_engaged[window]) and not np.any(off_engaged[window]))
  ok = trace_ok and td_ok and held
  print(f"[ON timed] latch@{latch_on_step} td@{td_step}(reported {td_reported}) "
        f"release@{release_step} held_past_gap={held} trace_exact={trace_ok} "
        f"-> {'PASS' if ok else 'FAIL'}")
  return ok


def main():
  if not os.path.exists(FIX):
    print(f"FAIL: fixture missing: {FIX}"); return 1
  fx = np.load(FIX, allow_pickle=False)
  seq = _inputs(fx)
  ok1 = test_off_byte_identical(fx, seq)
  ok2 = test_on_latches_past_gap(fx, seq)
  ok3 = test_on_timed_release(fx, seq)
  print(f"\n{'ALL PASS' if (ok1 and ok2 and ok3) else 'FAILURES PRESENT'}")
  return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
  sys.exit(main())

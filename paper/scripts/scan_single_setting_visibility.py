"""Verify the visibility-scan claims and produce the data for the
manuscript's rotated-BB84 figure (Sec. "Warm-up", Fig. 1).

Claims checked (rotated BB84, key from y* = 0 only, preparations
depolarized with visibility eta):

  [S1] The OPT adversary's exact guessing probability is AFFINE in eta:
       P_guess(eta) = 1 - (sqrt(2)/4) eta   (checked pointwise on a grid).
  [S2] Consequently the exact OPT key rate is
       r_LP(eta) = eta/sqrt(2) - h(1/2 + eta/(2 sqrt 2)),
       using the reverse-Fano bound H >= 2(1 - P_guess) (valid for
       P_guess >= 1/2) and the analytic error-correction cost.
  [S3] r_LP crosses zero at eta* ~ 0.9319 (bisection on the LP).
  [S4] The quantum adversary's certified rate (SDP) crosses zero near
       eta ~ 0.922 and reaches ~0.399 at eta = 1. The scan is run at
       level 2 of the Moroder-type hierarchy (npa_level_bob =
       npa_level_eve = 2) because at the default level 1 the relaxation
       is no stronger than the LP for eta < 1; levels only ever tighten
       the certified curve.

Writes paper/data/single_setting_visibility.json (consumed by
fig_single_setting.py).  Runtime: LP part seconds; SDP level-2 scan
~15-25 minutes.  Pass --lp-only to skip the SDP scan.
"""

import json
import math
import sys
import time
from pathlib import Path

import sympy as sp

from _protocols import SINGLE_SETTING_WHERE_KEY, check, protocol, rotated_bb84_scenario

OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)


def binary_entropy(p: float) -> float:
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def analytic_rate(eta: float) -> float:
    return eta / math.sqrt(2) - binary_entropy(0.5 + eta / (2 * math.sqrt(2)))


ok = True
print("[S1]/[S2] LP visibility scan vs analytic forms:")
lp_rows = []
for num in range(20, 41):  # eta = 0.500 ... 1.000, step 0.025
    eta = sp.Rational(num, 40)
    p = protocol(rotated_bb84_scenario(eta), where_key=SINGLE_SETTING_WHERE_KEY)
    g = float(p.eve_guess_master_key_by_y_lp[0])
    r = p.key_rate_per_key_run(method="lp")
    ok &= check(f"G_LP(eta={float(eta):.3f}) = 1 - sqrt(2) eta/4", g, 1 - math.sqrt(2) * float(eta) / 4, tol=1e-6)
    ok &= check(f"r_LP(eta={float(eta):.3f}) analytic", r, analytic_rate(float(eta)), tol=1e-6)
    lp_rows.append({"eta": float(eta), "lp_guess_y0": g, "lp_rate_key_run_RF": r,
                    "honest_cost": p.other_party_uncertainty_key_weighted})

print("\n[S3] LP threshold by bisection:")
lo, hi = sp.Rational(3788, 4096), sp.Rational(3892, 4096)
for _ in range(14):
    mid = (lo + hi) / 2
    r = protocol(rotated_bb84_scenario(mid), where_key=SINGLE_SETTING_WHERE_KEY).key_rate_per_key_run(method="lp")
    lo, hi = (lo, mid) if r > 0 else (mid, hi)
eta_star = float((lo + hi) / 2)
ok &= check("eta* (LP, reverse-Fano)", eta_star, 0.9319, tol=5e-4)

results = {"visibility_scan_lp": lp_rows, "lp_rf_threshold_eta": eta_star}

if "--lp-only" not in sys.argv:
    print("\n[S4] quantum-adversary scan (SDP, Moroder level 2):")
    sdp_rows = []
    grid = [k / 20 for k in range(10, 21)] + [0.92, 0.94, 0.96, 0.98, 0.99]
    for eta_f in sorted(set(grid)):
        eta = sp.nsimplify(eta_f, rational=True)
        t0 = time.time()
        p = protocol(rotated_bb84_scenario(eta), where_key=SINGLE_SETTING_WHERE_KEY, level=2)
        raw = float(p.eve_sdp_solver.eve_success_probability)
        rate = p.key_rate_per_key_run(method="sdp")
        sdp_rows.append({"eta": eta_f, "sdp2_guess_raw": raw,
                         "sdp2_guess_capped": p.eve_guess_master_key_sdp,
                         "sdp2_rate_key_run_RF": rate,
                         "lp_rate_key_run_RF": p.key_rate_per_key_run(method="lp")})
        print(f"  eta={eta_f:.2f} guess<={raw:.6f} rate>={rate:+.6f} ({time.time()-t0:.0f}s)", flush=True)
    results["visibility_scan_sdp_level2"] = sdp_rows
    ok &= check("[S4] certified quantum rate at eta=1", sdp_rows[-1]["sdp2_rate_key_run_RF"], 0.3989, tol=2e-3)
    ok &= sdp_rows[[r["eta"] for r in sdp_rows].index(0.92)]["sdp2_rate_key_run_RF"] < 0
    ok &= sdp_rows[[r["eta"] for r in sdp_rows].index(0.94)]["sdp2_rate_key_run_RF"] > 0
    print("  OK  quantum threshold lies in (0.92, 0.94)" if ok else "  (see failures above)")

(OUT / "single_setting_visibility.json").write_text(json.dumps(results, indent=1))
print("\nwrote", OUT / "single_setting_visibility.json")
print("ALL CLAIMS VERIFIED" if ok else "SOME CLAIMS FAILED")
raise SystemExit(0 if ok else 1)

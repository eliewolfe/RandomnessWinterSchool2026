"""Verify the claims of manuscript Sec. "Key from a noncontextuality-
inequality violation alone" (Fig. 3 and surrounding text), and produce
the data behind that figure.

Eve is constrained by the operational equivalences plus a single lower
bound w on the protocol's contextual-fraction witness (the L2-refined
dual of the contextual-fraction LP), instead of full data consistency.

Claims checked:
  [W1] Hexagon (aligned where_key): the OPT adversary's guessing
       probability is affine in the imposed violation, P_guess = 1 - w/4
       (witness normalized: noncontextual bound 0, observed value 1), so
       via the reverse-Fano bound H >= 2(1-P_guess) the rate is w/2, and
       the FULL rate 0.500 is recovered at the observed violation.
  [W2] Rotated BB84 (key at y*=0): P_guess = 1 - w/2, so the rate is
       w - h(cos^2(pi/8)); at the observed violation
       w_obs = sqrt(2) - 1 this gives P_guess = (3 - sqrt 2)/2 ~ 0.793 and
       rate ~ -0.187 < 0: the violation alone certifies NO key against
       the operational adversary.
  [W3] Rotated BB84, quantum adversary (SDP at Moroder level 2 --
       level 1 is looser here; levels only strengthen the bound):
       at w = w_obs, P_guess <= ~0.500, recovering the full-data quantum
       rate ~0.399; the certified rate crosses zero near 76% of the
       maximal violation.
  [W4] Self-consistency: constraining Eve by the eavesdropping LP's own
       dual witness at its observed value reproduces the full-data LP
       guessing probability and rate exactly.

Writes paper/data/witness_only.json (consumed by fig_witness_only.py).
Runtime: LP parts ~1 minute; the three level-2 SDP points ~5 minutes.
"""

import json
import math
from pathlib import Path

import numpy as np

from contextualityqkd.contextuality import NoncontextualityAssessment

from _protocols import (
    HEXAGON_ALIGNED_WHERE_KEY,
    SINGLE_SETTING_WHERE_KEY,
    check,
    hexagon_scenario,
    protocol,
    rotated_bb84_scenario,
)

OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)


def binary_entropy(p: float) -> float:
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def contextual_fraction_witness(scenario):
    assess = NoncontextualityAssessment(
        scenario, monotone="contextual_fraction", backend_solver="highs", qp_solver="clarabel"
    )
    alpha = np.asarray(assess.inequality["contextual_fraction"], dtype=float)
    sense = assess.inequality_sense["contextual_fraction"]
    observed = float(np.sum(alpha * np.asarray(scenario.data_numeric)))
    witness_sense = ">=" if sense == "<=" else "<="
    return alpha, observed, witness_sense


ok = True
results = {}

print("[W1] hexagon, OPT adversary under witness-only constraint:")
sc = hexagon_scenario()
p = protocol(sc, where_key=HEXAGON_ALIGNED_WHERE_KEY)
alpha, w_obs, wsense = contextual_fraction_witness(sc)
ok &= check("  observed witness value (contextual fraction)", w_obs, 1.0, tol=1e-6)
curve = []
for w in np.linspace(0.0, w_obs, 13):
    r = p.key_rate_from_witness(alpha, float(w), witness_sense=wsense, method="lp")
    ok &= check(f"  P_guess(w={w:.3f}) = 1 - w/4", r["eve_guess"], 1 - w / 4, tol=1e-6)
    ok &= check(f"  rate(w={w:.3f}) = w/2", r["key_rate_per_key_run"], w / 2, tol=1e-6)
    curve.append({"w": float(w), **r})
ok &= check("  full rate recovered at w_obs", curve[-1]["key_rate_per_key_run"],
            p.key_rate_per_key_run(method="lp"), tol=1e-6)
results["hexagon"] = {"observed": w_obs, "lp_curve": curve,
                      "full_rate_key_run": p.key_rate_per_key_run(method="lp")}

print("\n[W2] rotated BB84, OPT adversary under witness-only constraint:")
sc = rotated_bb84_scenario()
p = protocol(sc, where_key=SINGLE_SETTING_WHERE_KEY)
alpha, w_obs, wsense = contextual_fraction_witness(sc)
ok &= check("  observed witness value = sqrt(2) - 1", w_obs, math.sqrt(2) - 1, tol=1e-6)
cost = binary_entropy(math.cos(math.pi / 8) ** 2)
curve = []
for w in np.linspace(0.0, w_obs, 13):
    r = p.key_rate_from_witness(alpha, float(w), witness_sense=wsense, method="lp")
    ok &= check(f"  P_guess(w={w:.3f}) = 1 - w/2", r["eve_guess"], 1 - w / 2, tol=1e-6)
    ok &= check(f"  rate(w={w:.3f}) = w - h(cos^2 pi/8)", r["key_rate_per_key_run"], w - cost, tol=1e-6)
    curve.append({"w": float(w), **r})
ok &= check("  P_guess at w_obs = (3 - sqrt 2)/2", curve[-1]["eve_guess"], (3 - math.sqrt(2)) / 2, tol=1e-6)
ok &= curve[-1]["key_rate_per_key_run"] < 0
print(f"  {'OK ' if curve[-1]['key_rate_per_key_run'] < 0 else 'FAIL'} "
      f"rate at w_obs is negative: {curve[-1]['key_rate_per_key_run']:+.6f}")
results["rotated_bb84"] = {"observed": w_obs, "lp_curve": curve,
                           "full_rate_key_run": p.key_rate_per_key_run(method="lp"),
                           "honest_cost": cost}

print("\n[W3] rotated BB84, quantum adversary (SDP, Moroder level 2):")
p2 = protocol(sc, where_key=SINGLE_SETTING_WHERE_KEY, level=2)
sdp_points = []
for frac in (0.5, 0.75, 1.0):
    w = frac * w_obs
    r = p2.key_rate_from_witness(alpha, float(w), witness_sense=wsense, method="sdp")
    sdp_points.append({"w": float(w), **r})
    print(f"  w/w_obs={frac:.2f}: P_guess<={r['eve_guess']:.6f} rate>={r['key_rate_per_key_run']:+.6f}", flush=True)
ok &= check("  P_guess at w_obs", sdp_points[-1]["eve_guess"], 0.5, tol=2e-3)
ok &= check("  rate at w_obs (full-data quantum rate)", sdp_points[-1]["key_rate_per_key_run"], 0.399, tol=3e-3)
ok &= sdp_points[1]["key_rate_per_key_run"] < 0 < sdp_points[2]["key_rate_per_key_run"]
lo, hi = sdp_points[1], sdp_points[2]
w_star = lo["w"] - lo["key_rate_per_key_run"] * (hi["w"] - lo["w"]) / (
    hi["key_rate_per_key_run"] - lo["key_rate_per_key_run"])
ok &= check("  threshold as fraction of maximal violation", w_star / w_obs, 0.76, tol=0.02)
results["rotated_bb84"]["sdp_points_level2"] = sdp_points
results["rotated_bb84"]["sdp2_threshold_fraction"] = w_star / w_obs

print("\n[W4] self-consistency of the eavesdropping LP's own dual witness:")
solver = p.eve_master_key_lp_solver
coeffs = solver.guess_bound_coeffs_by_y[0]
value = float(np.sum(coeffs * np.asarray(sc.data_numeric)))
r = p.key_rate_from_witness(coeffs, value, witness_sense="<=", method="lp")
ok &= check("  witness-only P_guess = full-data P_guess", r["eve_guess"],
            p.eve_guess_master_key_average_y_lp, tol=1e-6)
ok &= check("  witness-only rate = full-data rate", r["key_rate_per_key_run"],
            p.key_rate_per_key_run(method="lp"), tol=1e-6)

(OUT / "witness_only.json").write_text(json.dumps(results, indent=1))
print("\nwrote", OUT / "witness_only.json")
print("ALL CLAIMS VERIFIED" if ok else "SOME CLAIMS FAILED")
raise SystemExit(0 if ok else 1)

"""Verify the (3,2)-PORAC visibility-scan claims and produce the data for
the manuscript's PORAC figure (Sec. "Separating quantum from operational
adversaries", Fig. 2).

Claims checked (preparations depolarized with visibility eta, trivial
where_key, Alice holding the master key):

  [Q1] The OPT adversary's exact guessing probability is AFFINE in eta:
       P_guess(eta) = 1 - eta/(2 sqrt 3)   (checked pointwise).
  [Q2] Hence the exact OPT rate is
       r_LP(eta) = eta/sqrt(3) - h(1/2 + eta/(2 sqrt 3)),
       via the reverse-Fano bound H >= 2(1 - P_guess); it is NEGATIVE for
       every eta in [0, 1]: no key against the operational adversary at
       any noise level.
  [Q3] The quantum adversary's certified rate (SDP, Moroder level 1:
       moment words {1, B, E, BE}, "NPA 1+ABE") crosses zero at
       eta* ~ 0.944 and reaches ~0.256 at eta = 1.

Writes paper/data/porac_visibility.json (consumed by fig_porac.py).
Runtime: ~5-10 minutes (one level-1 SDP per grid point; requires MOSEK).
"""

import json
import math
import time
from pathlib import Path

from _protocols import check, porac_scenario, protocol

OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)


def binary_entropy(p: float) -> float:
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def analytic_lp_rate(eta: float) -> float:
    return eta / math.sqrt(3) - binary_entropy(0.5 + eta / (2 * math.sqrt(3)))


ok = True
rows = []
grid = [k / 20 for k in range(10, 21)] + [0.91, 0.92, 0.93, 0.94, 0.96, 0.97, 0.98, 0.99]
print("[Q1]/[Q2]/[Q3] visibility scan:")
for eta in sorted(set(grid)):
    t0 = time.time()
    p = protocol(porac_scenario(eta=eta), where_key=None)
    g = p.eve_guess_master_key_average_y_lp
    r_lp = p.key_rate_per_key_run(method="lp")
    ok &= check(f"G_LP(eta={eta:.2f}) = 1 - eta/(2 sqrt 3)", g, 1 - eta / (2 * math.sqrt(3)), tol=1e-6)
    ok &= check(f"r_LP(eta={eta:.2f}) analytic", r_lp, analytic_lp_rate(eta), tol=1e-6)
    ok &= r_lp < 0
    g_sdp = float(p.eve_sdp_solver.eve_success_probability)
    r_sdp = p.key_rate_per_key_run(method="sdp")
    rows.append({"eta": eta, "lp_guess_avg": g, "lp_rate_key_run_RF": r_lp,
                 "sdp1_guess_raw": g_sdp, "sdp1_rate_key_run_RF": r_sdp,
                 "honest_cost": p.other_party_uncertainty_key_weighted})
    print(f"  eta={eta:.2f} lp_rate={r_lp:+.6f}(<0) sdp_guess<={g_sdp:.6f} sdp_rate>={r_sdp:+.6f} "
          f"({time.time()-t0:.0f}s)", flush=True)

by_eta = {r["eta"]: r for r in rows}
ok &= by_eta[0.94]["sdp1_rate_key_run_RF"] < 0 < by_eta[0.96]["sdp1_rate_key_run_RF"]
lo, hi = 0.94, 0.96
r_lo, r_hi = by_eta[lo]["sdp1_rate_key_run_RF"], by_eta[hi]["sdp1_rate_key_run_RF"]
eta_star = lo - r_lo * (hi - lo) / (r_hi - r_lo)
ok &= check("[Q3] quantum threshold eta*", eta_star, 0.944, tol=3e-3)
ok &= check("[Q3] quantum rate at eta=1", by_eta[1.0]["sdp1_rate_key_run_RF"], 0.2556, tol=2e-3)

(OUT / "porac_visibility.json").write_text(
    json.dumps({"eta_scan": rows, "sdp1_threshold_eta": eta_star}, indent=1)
)
print("\nwrote", OUT / "porac_visibility.json")
print("ALL CLAIMS VERIFIED" if ok else "SOME CLAIMS FAILED")
raise SystemExit(0 if ok else 1)

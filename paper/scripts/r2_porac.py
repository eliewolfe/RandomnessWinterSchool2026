"""R2: (3,2)-PORAC — OPT (LP) vs quantum (SDP) key rates, plus eta scan."""
import json, time
from pathlib import Path
import numpy as np
from contextualityqkd.demos.qkd_porac_3_2 import build_porac_scenario
from contextualityqkd.protocol import ContextualityProtocol

OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)

def proto(eta, level=1):
    return ContextualityProtocol(
        scenario=build_porac_scenario(eta=eta), where_key=None,
        master_key_holder="Alice", lp_solver="highs", sdp_solver="MOSEK",
        sdp_projective_bob=False, sdp_projective_eve=False,
        sdp_npa_level_bob=level, sdp_npa_level_eve=level,
        sdp_use_u_only=True, sdp_threads=1, sdp_verbose=0)

results = {}
p = proto(1.0)
t0 = time.time()
results["noiseless"] = {
    "lp_guess_by_y": np.asarray(p.eve_guess_master_key_by_y_lp).tolist(),
    "lp_guess_avg": p.eve_guess_master_key_average_y_lp,
    "honest_cost": p.other_party_uncertainty_key_weighted,
    "lp_rate_key_run_RF": p.key_rate_per_key_run(method="lp"),
    "lp_rate_key_run_ME": p.key_rate_per_key_run(method="lp", rate_type="min_entropy"),
    "sdp1_guess_raw": float(p.eve_sdp_solver.eve_success_probability),
    "sdp1_rate_key_run_RF": p.key_rate_per_key_run(method="sdp"),
    "sdp1_rate_key_run_ME": p.key_rate_per_key_run(method="sdp", rate_type="min_entropy"),
}
print("noiseless L1:", json.dumps(results["noiseless"], indent=1), f"({time.time()-t0:.0f}s)", flush=True)

t0 = time.time()
p2 = proto(1.0, level=2)
results["noiseless"]["sdp2_guess_raw"] = float(p2.eve_sdp_solver.eve_success_probability)
results["noiseless"]["sdp2_rate_key_run_RF"] = p2.key_rate_per_key_run(method="sdp")
print(f"noiseless L2: guess={results['noiseless']['sdp2_guess_raw']:.6f} rate={results['noiseless']['sdp2_rate_key_run_RF']:+.6f} ({time.time()-t0:.0f}s)", flush=True)

scan = []
for k in range(10, 21):
    eta = k / 20
    t0 = time.time()
    p = proto(eta)
    row = {"eta": eta,
           "lp_guess_avg": p.eve_guess_master_key_average_y_lp,
           "lp_rate_key_run_RF": p.key_rate_per_key_run(method="lp"),
           "sdp1_guess_raw": float(p.eve_sdp_solver.eve_success_probability),
           "sdp1_rate_key_run_RF": p.key_rate_per_key_run(method="sdp"),
           "honest_cost": p.other_party_uncertainty_key_weighted}
    scan.append(row)
    print(f"eta={eta:.2f} lp_G={row['lp_guess_avg']:.6f} lp_rate={row['lp_rate_key_run_RF']:+.6f} "
          f"sdp_G={row['sdp1_guess_raw']:.6f} sdp_rate={row['sdp1_rate_key_run_RF']:+.6f} ({time.time()-t0:.0f}s)", flush=True)
results["eta_scan"] = scan
(OUT / "r2_porac.json").write_text(json.dumps(results, indent=1))
print("saved r2_porac.json")

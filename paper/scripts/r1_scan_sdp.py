"""Refine R1: LP threshold bisection + sparse SDP visibility scan."""
import json
from pathlib import Path
import numpy as np
import sympy as sp
from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.quantum import GPTContextualityScenario

G = GPTContextualityScenario
OUT = Path(__file__).resolve().parents[1] / "data"
ANGLES_STATES = [0, sp.pi, sp.pi/2, -sp.pi/2]
ANGLES_MIS = [sp.pi/4, sp.pi + sp.pi/4, sp.pi/2 + sp.pi/4, -sp.pi/2 + sp.pi/4]

def hs(a): return G.projector_hs_vector(G.xz_plane_ket(a))

def scenario(eta):
    mixed = (hs(0) + hs(sp.pi)) / 2
    states = [sp.simplify(eta*hs(a) + (1-eta)*mixed) for a in ANGLES_STATES]
    return GPTContextualityScenario(gpt_states=states, gpt_effects=[hs(a) for a in ANGLES_MIS],
                                    measurement_indices=[(0,1),(2,3)], verbose=False)

def proto(eta):
    return ContextualityProtocol(scenario=scenario(eta), where_key=[(0,1,2,3),()],
        master_key_holder="Alice", lp_solver="highs", sdp_solver="MOSEK",
        sdp_projective_bob=False, sdp_projective_eve=False,
        sdp_npa_level_bob=1, sdp_npa_level_eve=1, sdp_use_u_only=True,
        sdp_threads=1, sdp_verbose=0)

# LP threshold bisection on rational grid denominator 4096
lo, hi = sp.Rational(3788, 4096), sp.Rational(3892, 4096)  # brackets 0.925..0.950
for _ in range(14):
    mid = (lo + hi) / 2
    r = proto(mid).key_rate_per_key_run(method="lp")
    if r > 0: hi = mid
    else: lo = mid
lp_threshold = float((lo + hi) / 2)
print("LP RF threshold eta* ≈", lp_threshold, flush=True)

sdp_scan = []
for num in range(10, 21):  # eta = 0.5 .. 1.0 step 0.05
    eta = sp.Rational(num, 20)
    p = proto(eta)
    row = {"eta": float(eta),
           "sdp_guess": p.eve_guess_master_key_sdp,
           "sdp_rate_key_run_RF": p.key_rate_per_key_run(method="sdp"),
           "lp_rate_key_run_RF": p.key_rate_per_key_run(method="lp")}
    sdp_scan.append(row)
    print(f"eta={row['eta']:.2f} sdp_G={row['sdp_guess']:.6f} sdp_rate={row['sdp_rate_key_run_RF']:+.6f}", flush=True)

res = json.loads((OUT/"r1_results.json").read_text())
res["lp_rf_threshold_eta"] = lp_threshold
res["visibility_scan_sdp"] = sdp_scan
(OUT/"r1_results.json").write_text(json.dumps(res, indent=1))
print("updated r1_results.json")

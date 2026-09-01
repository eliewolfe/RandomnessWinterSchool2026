"""R1 production runs: single-key-setting warm-up (Results VI.A).

Outputs paper/data/r1_results.json with:
- noiseless aligned-BB84 vs misaligned-XZ numbers (LP exact + SDP certified);
- LP rate-vs-visibility scan for the misaligned protocol keyed on y=0;
- the refined dual witness inequality for y=0.
"""
import json
from pathlib import Path

import numpy as np
import sympy as sp

from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.quantum import GPTContextualityScenario

G = GPTContextualityScenario
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)

ANGLES_STATES = [0, sp.pi, sp.pi / 2, -sp.pi / 2]          # |0>,|1>,|+>,|->
ANGLES_ALIGNED = ANGLES_STATES                              # Bob = same bases
ANGLES_MISALIGNED = [sp.pi/4, sp.pi + sp.pi/4, sp.pi/2 + sp.pi/4, -sp.pi/2 + sp.pi/4]

def hs(angle):
    return G.projector_hs_vector(G.xz_plane_ket(angle))

def depolarized_states(eta: sp.Rational):
    mixed = (hs(0) + hs(sp.pi)) / 2
    return [sp.simplify(eta * hs(a) + (1 - eta) * mixed) for a in ANGLES_STATES]

def scenario_for(effect_angles, eta=sp.Integer(1)):
    states = depolarized_states(eta)
    effects = [hs(a) for a in effect_angles]
    return GPTContextualityScenario(
        gpt_states=states, gpt_effects=effects,
        measurement_indices=[(0, 1), (2, 3)], verbose=False,
    )

def protocol_for(scenario, where_key, with_sdp=False):
    return ContextualityProtocol(
        scenario=scenario, where_key=where_key, master_key_holder="Alice",
        lp_solver="highs", sdp_solver="MOSEK",
        sdp_projective_bob=False, sdp_projective_eve=False,
        sdp_npa_level_bob=1, sdp_npa_level_eve=1,
        sdp_use_u_only=True, sdp_threads=1, sdp_verbose=0,
    )

def summarize(protocol, with_sdp):
    out = {
        "lp_guess_by_y": np.asarray(protocol.eve_guess_master_key_by_y_lp).tolist(),
        "lp_guess_avg": protocol.eve_guess_master_key_average_y_lp,
        "honest_cost_H(K|B)": protocol.other_party_uncertainty_key_weighted,
        "lp_rate_key_run_RF": protocol.key_rate_per_key_run(method="lp"),
        "lp_rate_key_run_ME": protocol.key_rate_per_key_run(method="lp", rate_type="min_entropy"),
        "lp_rate_exp_run_RF": protocol.key_rate_per_experimental_run(method="lp"),
        "key_gen_prob": protocol.key_generation_probability_per_run,
    }
    if with_sdp:
        out["sdp_guess"] = protocol.eve_guess_master_key_sdp
        out["sdp_rate_key_run_RF"] = protocol.key_rate_per_key_run(method="sdp")
        out["sdp_rate_exp_run_RF"] = protocol.key_rate_per_experimental_run(method="sdp")
    return out

results = {}
single_y0 = [(0, 1, 2, 3), ()]

print("== aligned BB84, key from y=0 only ==", flush=True)
p = protocol_for(scenario_for(ANGLES_ALIGNED), single_y0)
results["aligned_single_y0"] = summarize(p, with_sdp=True)
print(json.dumps(results["aligned_single_y0"], indent=1), flush=True)

print("== misaligned XZ, key from y=0 only ==", flush=True)
p = protocol_for(scenario_for(ANGLES_MISALIGNED), single_y0)
results["misaligned_single_y0"] = summarize(p, with_sdp=True)
print(json.dumps(results["misaligned_single_y0"], indent=1), flush=True)
witness = p.eve_master_key_lp_solver.guess_bound_coeffs_by_y[0]
results["misaligned_witness_y0_coeffs"] = np.asarray(witness).tolist()
results["misaligned_witness_y0_text"] = p.format_eve_guess_upper_bound_inequality_by_y(precision=4)

print("== visibility scan (LP), misaligned, key from y=0 ==", flush=True)
scan = []
for num in range(20, 41):  # eta = 0.50 ... 1.00 in steps of 0.025
    eta = sp.Rational(num, 40)
    p = protocol_for(scenario_for(ANGLES_MISALIGNED, eta), single_y0)
    scan.append({
        "eta": float(eta),
        "lp_guess_y0": float(p.eve_guess_master_key_by_y_lp[0]),
        "lp_rate_key_run_RF": p.key_rate_per_key_run(method="lp"),
        "lp_rate_key_run_ME": p.key_rate_per_key_run(method="lp", rate_type="min_entropy"),
        "honest_cost": p.other_party_uncertainty_key_weighted,
    })
    print(f"eta={float(eta):.3f} G={scan[-1]['lp_guess_y0']:.6f} rate_RF={scan[-1]['lp_rate_key_run_RF']:+.6f}", flush=True)
results["visibility_scan_lp"] = scan

(OUT / "r1_results.json").write_text(json.dumps(results, indent=1))
print("saved", OUT / "r1_results.json")

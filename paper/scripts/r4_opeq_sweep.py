"""R4: Eve LP power under full vs completeness-only measurement OPEQs."""
import json
from pathlib import Path
import numpy as np
import sympy as sp
from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.quantum import GPTContextualityScenario
from contextualityqkd.demos.qkd_porac_3_2 import build_porac_scenario

OUT = Path(__file__).resolve().parents[1] / "data"
G = GPTContextualityScenario

def ring(ns, nm):
    half = ns // 2
    bases = sorted(set((m * half) // nm for m in range(nm)))
    return G.from_xz_ring(num_states=ns, measurement_indices=tuple((i, i+half) for i in bases), verbose=False)

def misaligned():
    states = [G.projector_hs_vector(G.xz_plane_ket(k*sp.pi/2)) for k in range(4)]
    effects = [G.projector_hs_vector(G.xz_plane_ket(k*sp.pi/2 + sp.pi/4)) for k in range(4)]
    return GPTContextualityScenario(gpt_states=states, gpt_effects=effects,
                                    measurement_indices=[(0,2),(1,3)], verbose=False)

def bb84():
    kets = [G.xz_plane_ket(a) for a in (0, sp.pi, sp.pi/2, -sp.pi/2)]
    vecs = [G.projector_hs_vector(k) for k in kets]
    return GPTContextualityScenario(gpt_states=vecs, gpt_effects=vecs,
                                    measurement_indices=[(0,1),(2,3)], verbose=False)

def cabello():
    labels = list("123456789ABCDEFGHI")
    rays = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[1,1,1,1],[1,-1,1,-1],[1,-1,-1,1],
                     [1,-1,-1,-1],[1,-1,1,1],[1,1,1,-1],[1,1,0,0],[0,0,1,1],[0,0,1,-1],
                     [0,1,0,1],[0,1,0,-1],[1,0,-1,0],[1,0,0,-1],[1,0,0,1],[0,1,-1,0]], dtype=int)
    contexts = ["12BC","13DE","23GH","45EF","46GI","56AB","78AC","79HI","89DF"]
    l2i = {lab: i for i, lab in enumerate(labels)}
    mi = [tuple(l2i[ch] for ch in c) for c in contexts]
    return G.from_integer_rays(rays=rays, measurement_indices=mi, verbose=False)

CASES = {
    "bb84_aligned": (bb84, None),
    "xz_misaligned": (misaligned, None),
    "hexagon": (lambda: ring(6, 3), [(0, 3), (1, 4), (2, 5)]),
    "xz_ring_8s_4m": (lambda: ring(8, 4), None),
    "porac_3_2": (lambda: build_porac_scenario(eta=1.0), None),
    "cabello_18ray": (cabello, None),
}

results = {}
for name, (build, wk) in CASES.items():
    sc_full = build()
    sc_rest = sc_full.restricted_to_completeness_meas_opeqs()
    row = {"num_meas_opeqs_full": int(sc_full.opeq_meas_numeric.shape[0]),
           "num_meas_opeqs_restricted": int(sc_rest.opeq_meas_numeric.shape[0])}
    for label, sc in [("full", sc_full), ("completeness_only", sc_rest)]:
        p = ContextualityProtocol(scenario=sc, where_key=wk, master_key_holder="Alice", lp_solver="highs")
        row[label] = {
            "lp_guess_avg": p.eve_guess_master_key_average_y_lp,
            "rate_key_run_RF": p.key_rate_per_key_run(method="lp"),
            "rate_key_run_ME": p.key_rate_per_key_run(method="lp", rate_type="min_entropy"),
        }
    results[name] = row
    f, r = row["full"], row["completeness_only"]
    print(f"{name:16s} opeqs {row['num_meas_opeqs_full']:3d}->{row['num_meas_opeqs_restricted']:2d} | "
          f"G: {f['lp_guess_avg']:.6f} -> {r['lp_guess_avg']:.6f} | "
          f"rate_RF: {f['rate_key_run_RF']:+.6f} -> {r['rate_key_run_RF']:+.6f}", flush=True)

(OUT / "r4_opeq_sweep.json").write_text(json.dumps(results, indent=1))
print("saved r4_opeq_sweep.json")

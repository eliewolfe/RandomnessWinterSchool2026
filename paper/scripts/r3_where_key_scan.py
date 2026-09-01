"""R3: trivial vs automatically optimized where_key across candidate scenarios (LP)."""
import json
from pathlib import Path
import numpy as np
import sympy as sp
from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.quantum import GPTContextualityScenario

OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)
G = GPTContextualityScenario

def ring(num_states, num_meas):
    half = num_states // 2
    bases = sorted(set((m * half) // num_meas for m in range(num_meas)))
    idx = tuple((i, i + half) for i in bases)
    return G.from_xz_ring(num_states=num_states, measurement_indices=idx, verbose=False)

def misaligned():
    states = [G.projector_hs_vector(G.xz_plane_ket(k*sp.pi/2)) for k in range(4)]
    effects = [G.projector_hs_vector(G.xz_plane_ket(k*sp.pi/2 + sp.pi/4)) for k in range(4)]
    return GPTContextualityScenario(gpt_states=states, gpt_effects=effects,
                                    measurement_indices=[(0,2),(1,3)], verbose=False)

def cabello():
    labels = list("123456789ABCDEFGHI")
    rays = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[1,1,1,1],[1,-1,1,-1],[1,-1,-1,1],
                     [1,-1,-1,-1],[1,-1,1,1],[1,1,1,-1],[1,1,0,0],[0,0,1,1],[0,0,1,-1],
                     [0,1,0,1],[0,1,0,-1],[1,0,-1,0],[1,0,0,-1],[1,0,0,1],[0,1,-1,0]], dtype=int)
    contexts = ["12BC","13DE","23GH","45EF","46GI","56AB","78AC","79HI","89DF"]
    l2i = {lab: i for i, lab in enumerate(labels)}
    mi = [tuple(l2i[ch] for ch in c) for c in contexts]
    return G.from_integer_rays(rays=rays, measurement_indices=mi, verbose=False)

CANDIDATES = {
    "xz_ring_8s_4m": lambda: ring(8, 4),
    "xz_ring_12s_6m": lambda: ring(12, 6),
    "xz_ring_16s_8m": lambda: ring(16, 8),
    "hexagon_6s_3m": lambda: ring(6, 3),
    "xz_misaligned_4s_2m": misaligned,
    "cabello_18ray": cabello,
}

results = {}
for name, build in CANDIDATES.items():
    sc = build()
    row = {}
    for label, wk in [("trivial", None), ("auto", "Automatic")]:
        p = ContextualityProtocol(scenario=sc, where_key=wk, master_key_holder="Alice",
                                  lp_solver="highs", optimize_verbose=False)
        row[label] = {
            "where_key": [list(r) for r in p.where_key],
            "key_gen_prob": p.key_generation_probability_per_run,
            "rate_key_run_RF": p.key_rate_per_key_run(method="lp"),
            "rate_exp_run_RF": p.key_rate_per_experimental_run(method="lp"),
            "rate_key_run_ME": p.key_rate_per_key_run(method="lp", rate_type="min_entropy"),
            "rate_exp_run_ME": p.key_rate_per_experimental_run(method="lp", rate_type="min_entropy"),
        }
    results[name] = row
    t, a = row["trivial"], row["auto"]
    print(f"{name:22s} trivial exp_RF={t['rate_exp_run_RF']:+.6f} | auto exp_RF={a['rate_exp_run_RF']:+.6f} "
          f"(keyfrac {a['key_gen_prob']:.3f}) auto_wk={a['where_key']}", flush=True)

(OUT / "r3_where_key.json").write_text(json.dumps(results, indent=1))
print("saved r3_where_key.json")

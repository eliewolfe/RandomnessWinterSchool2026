"""R1 exploration: single-key-setting protocols, LP (OPT adversary) numbers."""
import numpy as np
import sympy as sp
from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.quantum import GPTContextualityScenario

G = GPTContextualityScenario

def bb84_aligned():
    kets = [G.xz_plane_ket(a) for a in (0, sp.pi, sp.pi/2, -sp.pi/2)]
    vecs = [G.projector_hs_vector(k) for k in kets]
    return GPTContextualityScenario(gpt_states=vecs, gpt_effects=vecs,
                                    measurement_indices=[(0,1),(2,3)], verbose=False)

def xz_misaligned():
    states = [G.projector_hs_vector(G.xz_plane_ket(k*sp.pi/2)) for k in range(4)]
    effects = [G.projector_hs_vector(G.xz_plane_ket(k*sp.pi/2 + sp.pi/4)) for k in range(4)]
    return GPTContextualityScenario(gpt_states=states, gpt_effects=effects,
                                    measurement_indices=[(0,2),(1,3)], verbose=False)

def report(name, scenario, where_key, label):
    p = ContextualityProtocol(scenario=scenario, where_key=where_key,
                              master_key_holder="Alice", lp_solver="highs")
    g = p.eve_guess_master_key_by_y_lp
    print(f"{name:14s} {label:22s} G_E(y)={np.round(g,6)} "
          f"rate/keyrun[RF]={p.key_rate_per_key_run(method='lp'):+.6f} "
          f"rate/exprun[RF]={p.key_rate_per_experimental_run(method='lp'):+.6f} "
          f"rate/keyrun[ME]={p.key_rate_per_key_run(method='lp', rate_type='min_entropy'):+.6f}")

for name, sc in [("BB84-aligned", bb84_aligned()), ("XZ-misaligned", xz_misaligned())]:
    nx = sc.X_cardinality
    report(name, sc, None, "all y keyed")
    report(name, sc, [tuple(range(nx)), ()], "key only from y=0")
    report(name, sc, [(), tuple(range(nx))], "key only from y=1")

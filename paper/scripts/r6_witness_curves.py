"""R6: key rate from a noncontextuality-inequality violation alone."""
import json
from pathlib import Path
import numpy as np
import sympy as sp
from contextualityqkd.contextuality import NoncontextualityAssessment
from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.quantum import GPTContextualityScenario

OUT = Path(__file__).resolve().parents[1] / "data"
G = GPTContextualityScenario

def misaligned():
    states = [G.projector_hs_vector(G.xz_plane_ket(k*sp.pi/2)) for k in range(4)]
    effects = [G.projector_hs_vector(G.xz_plane_ket(k*sp.pi/2 + sp.pi/4)) for k in range(4)]
    return GPTContextualityScenario(gpt_states=states, gpt_effects=effects,
                                    measurement_indices=[(0,2),(1,3)], verbose=False)

sc = misaligned()
protocol = ContextualityProtocol(scenario=sc, where_key=[(0,1,2,3),()],
                                 master_key_holder="Alice", lp_solver="highs")

assess = NoncontextualityAssessment(sc, monotone="contextual_fraction",
                                    backend_solver="highs", qp_solver="clarabel")
alpha = np.asarray(assess.inequality["contextual_fraction"], dtype=float)
nc_bound = float(assess.inequality_bound["contextual_fraction"])
sense = assess.inequality_sense["contextual_fraction"]
observed = float(np.sum(alpha * np.asarray(sc.data_numeric)))
print(f"NC inequality: sense={sense} nc_bound={nc_bound:.6f} observed={observed:.6f} "
      f"violation={assess.violation['contextual_fraction']:.6f}", flush=True)
witness_sense = ">=" if sense == "<=" else "<="

full_rate = protocol.key_rate_per_key_run(method="lp")
print(f"full-data LP rate/keyrun = {full_rate:+.6f}", flush=True)

curve = []
lo, hi = (nc_bound, observed) if witness_sense == ">=" else (observed, nc_bound)
ws = np.linspace(nc_bound, observed, 13)
for w in ws:
    r = protocol.key_rate_from_witness(alpha, float(w), witness_sense=witness_sense, method="lp")
    curve.append({"w": float(w), **r})
    print(f"w={w:.6f} guess={r['eve_guess']:.6f} rate_key={r['key_rate_per_key_run']:+.6f}", flush=True)

# SDP points at the observed violation and midway
sdp_points = []
proto_sdp = ContextualityProtocol(scenario=sc, where_key=[(0,1,2,3),()],
    master_key_holder="Alice", lp_solver="highs", sdp_solver="MOSEK",
    sdp_projective_bob=False, sdp_projective_eve=False,
    sdp_npa_level_bob=2, sdp_npa_level_eve=2, sdp_use_u_only=True, sdp_threads=2, sdp_verbose=0)
for w in [float(ws[6]), float(ws[9]), float(ws[12])]:
    r = proto_sdp.key_rate_from_witness(alpha, w, witness_sense=witness_sense, method="sdp")
    sdp_points.append({"w": w, **r})
    print(f"SDP L2 w={w:.6f} guess={r['eve_guess']:.6f} rate_key={r['key_rate_per_key_run']:+.6f}", flush=True)

json.dump({"alpha": alpha.tolist(), "nc_bound": nc_bound, "observed": observed,
           "sense": sense, "witness_sense": witness_sense, "full_rate_key_run": full_rate,
           "lp_curve": curve, "sdp_points_level2": sdp_points},
          open(OUT/"r6_witness.json", "w"), indent=1)
print("saved r6_witness.json")

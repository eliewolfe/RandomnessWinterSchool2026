"""R1: level-2 SDP visibility scan (quantum-adversary curve for F1)."""
import json, time
from pathlib import Path
import sympy as sp
from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.quantum import GPTContextualityScenario
G = GPTContextualityScenario
OUT = Path(__file__).resolve().parents[1] / "data"
AS = [0, sp.pi, sp.pi/2, -sp.pi/2]; AM = [a + sp.pi/4 for a in AS]
def hs(a): return G.projector_hs_vector(G.xz_plane_ket(a))
def scen(eta):
    mixed = (hs(0)+hs(sp.pi))/2
    return GPTContextualityScenario(gpt_states=[sp.simplify(eta*hs(a)+(1-eta)*mixed) for a in AS],
        gpt_effects=[hs(a) for a in AM], measurement_indices=[(0,1),(2,3)], verbose=False)
rows = []
for num in range(10, 21):
    eta = sp.Rational(num, 20)
    t0 = time.time()
    p = ContextualityProtocol(scenario=scen(eta), where_key=[(0,1,2,3),()],
        master_key_holder="Alice", lp_solver="highs", sdp_solver="MOSEK",
        sdp_projective_bob=False, sdp_projective_eve=False,
        sdp_npa_level_bob=2, sdp_npa_level_eve=2, sdp_use_u_only=True,
        sdp_threads=1, sdp_verbose=0)
    raw = float(p.eve_sdp_solver.eve_success_probability)
    rows.append({"eta": float(eta), "sdp2_guess_raw": raw,
                 "sdp2_guess_capped": p.eve_guess_master_key_sdp,
                 "sdp2_rate_key_run_RF": p.key_rate_per_key_run(method="sdp"),
                 "lp_rate_key_run_RF": p.key_rate_per_key_run(method="lp")})
    print(f"eta={rows[-1]['eta']:.2f} raw={raw:.6f} rate={rows[-1]['sdp2_rate_key_run_RF']:+.6f} ({time.time()-t0:.0f}s)", flush=True)
res = json.loads((OUT/"r1_results.json").read_text())
res["visibility_scan_sdp_level2"] = rows
(OUT/"r1_results.json").write_text(json.dumps(res, indent=1))
print("updated")

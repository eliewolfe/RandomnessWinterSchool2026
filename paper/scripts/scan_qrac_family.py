"""Verify the random-access-code family claims of manuscript Sec.
"Random access codes: key rate versus dimension and word count", and
produce the data behind its table and figure.

Part A -- plain (2,d) QRACs, d = 2..8. Alice sends the proven-optimal
QRAC states (top eigenvectors of the summed MUB projectors); Bob measures
the computational or Fourier basis; the key dit is the addressed letter
(where_key = all). Claims:
  [A1] Bob's success equals (1 + 1/sqrt(d))/2, Eq. (45) of Ambainis et
       al. [arXiv:1510.03045] (optimality: Farkas-Kaniewski).
  [A2] For every odd d the optimal states are linearly INDEPENDENT: the
       ensemble supports no preparation operational equivalence, and the
       exact operational-adversary guessing probability is 1 -- no key,
       and (by CFW Observation 3 / Baldijao et al.) no contextuality at
       all. Even d retain a few accidental equivalences but no positive
       rate either. Only d = 2 (the rotated-BB84 protocol of Results A)
       certifies key.

Part B -- d-ary oblivious multiplexing (2,d)-POM, d = 2..6 [Baldijao et
al., arXiv:2607.26145, Example 1, hiding every direction a with all
components nonzero]. States and POVMs from a convex see-saw (states step
is a single SDP for fixed measurements); promised preparation
equivalences = exactly the imposed obliviousness relations; measurement
equivalences = effect-level nullspace. Claims:
  [B1] d = 2 recovers the Spekkens et al. parity-oblivious optimum
       (1 + 1/sqrt 2)/2 and reproduces the rotated-BB84 rates.
  [B2] For d >= 3 obliviousness is costly (see-saw success strictly below
       the unconstrained QRAC value), and the resulting key rates -- LP
       exact and SDP (Moroder level 1) certified -- are reported per d.

Writes paper/data/qrac_family.json (consumed by fig_qrac_family.py).
Runtime: ~20-40 minutes total (see-saws are seconds; the level-1 SDPs
grow with d; the largest SDP included is d = 5).
"""

import json
import math
import time
from pathlib import Path

import numpy as np

from _protocols import (
    check,
    dary_parity_directions,
    pom_seesaw_states_and_measurements,
    protocol,
    qrac_optimal_states,
    qrac_words,
    rac_scenario_from_states,
)

OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)

ok = True
results = {"plain": [], "pom": []}

print("Part A: plain (2,d) MUB-QRACs, where_key = all")
for d in range(2, 9):
    states, bases = qrac_optimal_states(2, d)
    povms = [[np.outer(b[:, k], b[:, k].conj()) for k in range(d)] for b in bases]
    sc = rac_scenario_from_states(states, povms, 2, d)
    p = protocol(sc, where_key=None)
    pbob = float(np.mean(np.max(np.asarray(sc.data_numeric), axis=2)))
    ok &= check(f"[A1] (2,{d}) Bob success = (1+1/sqrt(d))/2", pbob, 0.5 * (1 + 1 / math.sqrt(d)), tol=1e-6)
    n_opeq = int(sc.opeq_preps_numeric.shape[0])
    g = p.eve_guess_master_key_average_y_lp
    row = {"d": d, "bob_success": pbob, "n_prep_opeqs": n_opeq,
           "lp_guess": g, "lp_rate_key_run_RF": p.key_rate_per_key_run(method="lp"),
           "honest_cost": p.other_party_uncertainty_key_weighted}
    results["plain"].append(row)
    print(f"  d={d}: opeqs={n_opeq:2d} G_LP={g:.4f} rate_RF={row['lp_rate_key_run_RF']:+.4f}", flush=True)
    if d % 2 == 1:
        ok &= check(f"[A2] odd d={d}: no prep opeqs", n_opeq, 0, tol=0.1)
        ok &= check(f"[A2] odd d={d}: Eve guesses perfectly", g, 1.0, tol=1e-6)
ok &= all(r["lp_rate_key_run_RF"] < 1e-9 for r in results["plain"] if r["d"] > 2)
print("  OK  [A2] no plain (2,d>2) QRAC certifies key against the OPT adversary")

print("\nPart B: d-ary oblivious multiplexing (2,d)-POM")
for d in range(2, 7):
    t0 = time.time()
    success, states, povms = pom_seesaw_states_and_measurements(2, d, rounds=8)
    dirs = dary_parity_directions(2, d)
    sc = rac_scenario_from_states(states, povms, 2, d, promised_dirs=dirs)
    keymap_ok = all(int(sc.key_selection_by_xy[i, y]) == x[y]
                    for i, x in enumerate(qrac_words(2, d)) for y in range(2))
    p = protocol(sc, where_key=None)
    g_lp = p.eve_guess_master_key_average_y_lp
    row = {"d": d, "n_hidden_dirs": len(dirs), "pom_success": success,
           "unconstrained_success": 0.5 * (1 + 1 / math.sqrt(d)),
           "keymap_ok": keymap_ok,
           "lp_guess": g_lp,
           "lp_rate_key_run_RF": p.key_rate_per_key_run(method="lp"),
           "honest_cost": p.other_party_uncertainty_key_weighted}
    print(f"  d={d}: S_POM={success:.5f} (unconstrained {row['unconstrained_success']:.5f}) "
          f"G_LP={g_lp:.5f} rate_LP={row['lp_rate_key_run_RF']:+.5f} keymap_ok={keymap_ok} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if d <= 5:
        t0 = time.time()
        raw = float(p.eve_sdp_solver.eve_success_probability)
        row["sdp1_guess_raw"] = raw
        row["sdp1_rate_key_run_RF"] = p.key_rate_per_key_run(method="sdp")
        row["sdp1_rate_key_run_ME"] = p.key_rate_per_key_run(method="sdp", rate_type="min_entropy")
        print(f"        SDP(Moroder L1): guess<={raw:.5f} rate_RF>={row['sdp1_rate_key_run_RF']:+.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    results["pom"].append(row)

b2 = results["pom"][0]
ok &= check("[B1] d=2 POM success = (1+1/sqrt 2)/2", b2["pom_success"], 0.5 * (1 + 1 / math.sqrt(2)), tol=1e-4)
ok &= check("[B1] d=2 POM LP rate = rotated-BB84 rate", b2["lp_rate_key_run_RF"], 0.106231, tol=1e-3)
for r in results["pom"]:
    if r["d"] >= 3:
        ok &= r["pom_success"] < r["unconstrained_success"] - 1e-4
print("  OK  [B2] obliviousness is costly for every d >= 3" if ok else "  (see failures)")

(OUT / "qrac_family.json").write_text(json.dumps(results, indent=1))
print("\nwrote", OUT / "qrac_family.json")
print("ALL CLAIMS VERIFIED" if ok else "SOME CLAIMS FAILED")
raise SystemExit(0 if ok else 1)

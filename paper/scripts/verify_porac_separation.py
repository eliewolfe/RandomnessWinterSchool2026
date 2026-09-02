"""Verify the noiseless (3,2)-PORAC claims of manuscript Sec.
"Separating quantum from operational adversaries" (trivial where_key,
Alice holding the master key).

Claims checked:
  [P1] scenario shape: X=8, Y=3, B=2; 4 preparation (parity-obliviousness)
       operational equivalences; key map k(x,y) = y-th bit of x.
  [P2] optimal quantum decoding probability (from the data) is
       (1 + 1/sqrt(3))/2 ~ 0.7887, with error-correction cost
       H(K|B,y) = h((1 + 1/sqrt 3)/2) ~ 0.744008.
  [P3] OPT adversary (LP, exact): P_guess(y) = 1 - 1/(2 sqrt 3) ~ 0.711325
       for every y.
  [P4] OPT reverse-Fano key rate = 2(1-P_guess) - cost ~ -0.166657 < 0:
       no key against the operational adversary.
  [P5] OPT min-entropy key rate ~ -0.252588 < 0.
  [P6] quantum adversary (SDP, Moroder level 1 -- moment words
       {1, B, E, BE}, "NPA 1+ABE"): P_guess <= ~0.5002.
  [P7] certified quantum reverse-Fano rate ~ +0.2556 per key round
       (min-entropy: ~ +0.2554).

Runtime: ~1 minute (requires MOSEK).
"""

import math

import numpy as np

from _protocols import check, porac_scenario, protocol


def binary_entropy(p: float) -> float:
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


ok = True
sc = porac_scenario(eta=1.0)

print("[P1] scenario structure:")
ok &= (sc.X_cardinality, sc.Y_cardinality, sc.B_cardinality) == (8, 3, 2)
ok &= sc.opeq_preps_numeric.shape[0] == 4
print(f"  {'OK ' if ok else 'FAIL'} X,Y,B = {(sc.X_cardinality, sc.Y_cardinality, sc.B_cardinality)}, "
      f"{sc.opeq_preps_numeric.shape[0]} preparation opeqs")
key_map_ok = all(
    int(sc.key_selection_by_xy[x, y]) == (x >> (2 - y)) % 2
    for x in range(8) for y in range(3)
)
print(f"  {'OK ' if key_map_ok else 'FAIL'} key map k(x,y) = y-th bit of x")
ok &= key_map_ok

p = protocol(sc, where_key=None)
print("\n[P2] honest-party quantities:")
bob_success = float(np.mean(np.max(np.asarray(sc.data_numeric), axis=2)))
ok &= check("Bob decoding probability", bob_success, (1 + 1 / math.sqrt(3)) / 2, tol=1e-6)
ok &= check("EC cost H(K|B,y)", p.other_party_uncertainty_key_weighted,
            binary_entropy((1 + 1 / math.sqrt(3)) / 2), tol=1e-6)

print("\n[P3]-[P5] operational adversary (LP, exact):")
for y, g in enumerate(p.eve_guess_master_key_by_y_lp):
    ok &= check(f"P_guess(y={y}) = 1 - 1/(2 sqrt 3)", float(g), 1 - 1 / (2 * math.sqrt(3)), tol=1e-6)
ok &= check("[P4] reverse-Fano rate/key run", p.key_rate_per_key_run(method="lp"), -0.166657, tol=1e-5)
ok &= check("[P5] min-entropy rate/key run",
            p.key_rate_per_key_run(method="lp", rate_type="min_entropy"), -0.252588, tol=1e-5)

print("\n[P6]-[P7] quantum adversary (SDP, Moroder level 1):")
ok &= check("[P6] P_guess bound", float(p.eve_sdp_solver.eve_success_probability), 0.5002, tol=1e-3)
ok &= check("[P7] reverse-Fano rate/key run", p.key_rate_per_key_run(method="sdp"), 0.2556, tol=2e-3)
ok &= check("[P7] min-entropy rate/key run",
            p.key_rate_per_key_run(method="sdp", rate_type="min_entropy"), 0.2554, tol=2e-3)

print("\nALL CLAIMS VERIFIED" if ok else "\nSOME CLAIMS FAILED")
raise SystemExit(0 if ok else 1)

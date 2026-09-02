"""Verify every noiseless numerical claim of manuscript Sec. "Warm-up:
key from a single measurement setting" (Table I and surrounding text).

Claims checked (all at visibility eta = 1, key generated at y* = 0 only,
Alice holding the master key):

Aligned BB84 (Bob measures Z and X):
  [A1] OPT adversary (LP, exact):      P_guess = 3/4
  [A2] reverse-Fano key rate:          0 bits per key round
  [A3] min-entropy key rate:           negative
  [A4] quantum adversary (SDP, Moroder level 1): P_guess bound = 3/4

Rotated BB84 (Bob measures the pi/4-rotated intermediate bases):
  [R1] OPT adversary (LP, exact):      P_guess = 1 - sqrt(2)/4 ~ 0.646447
  [R2] error-correction cost:          H(K|B,y*) = h(cos^2(pi/8)) ~ 0.600876
  [R3] reverse-Fano rate per key round: 2(1-P_guess) - cost ~ 0.106231
  [R4] rate per experimental run:      half of [R3] ~ 0.053115
  [R5] min-entropy rate per key round: ~ 0.028521
  [R6] quantum adversary (SDP, Moroder level 1): P_guess <= ~0.5001
  [R7] certified quantum rate:         ~ 0.398852 (~= 1 - cost = 0.399124)
  [R8] the y*=0 dual witness reads
       P_guess <= 1/2 + (1/4)[P(0|0,1)+P(1|1,1)+P(1|2,1)+P(0|3,1)],
       tight at the data where each bracketed term is sin^2(pi/8).

Runtime: ~1 minute. Requires MOSEK; the SDP uses npa_level_bob =
npa_level_eve = 1, i.e. moment words {1, B, E, BE}: level 1 of the
Moroder-type hierarchy ("NPA 1+ABE").
"""

import math

import numpy as np

from _protocols import (
    SINGLE_SETTING_WHERE_KEY,
    aligned_bb84_scenario,
    check,
    protocol,
    rotated_bb84_scenario,
)


def binary_entropy(p: float) -> float:
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


ok = True

print("Aligned BB84, key from y*=0 only:")
p = protocol(aligned_bb84_scenario(), where_key=SINGLE_SETTING_WHERE_KEY)
ok &= check("[A1] LP P_guess", p.eve_guess_master_key_average_y_lp, 3 / 4, tol=1e-6)
ok &= check("[A2] LP reverse-Fano rate/key run", p.key_rate_per_key_run(method="lp"), 0.0, tol=1e-6)
me = p.key_rate_per_key_run(method="lp", rate_type="min_entropy")
print(f"  {'OK ' if me < 0 else 'FAIL'} [A3] LP min-entropy rate is negative: {me:+.6f}")
ok &= me < 0
ok &= check("[A4] SDP (Moroder L1) P_guess bound", float(p.eve_sdp_solver.eve_success_probability), 3 / 4, tol=1e-3)

print("\nRotated BB84, key from y*=0 only:")
p = protocol(rotated_bb84_scenario(), where_key=SINGLE_SETTING_WHERE_KEY)
g_lp = p.eve_guess_master_key_average_y_lp
cost = p.other_party_uncertainty_key_weighted
ok &= check("[R1] LP P_guess = 1 - sqrt(2)/4", g_lp, 1 - math.sqrt(2) / 4, tol=1e-6)
ok &= check("[R2] EC cost = h(cos^2(pi/8))", cost, binary_entropy(math.cos(math.pi / 8) ** 2), tol=1e-6)
ok &= check("[R3] LP reverse-Fano rate/key run", p.key_rate_per_key_run(method="lp"), 2 * (1 - g_lp) - cost, tol=1e-6)
ok &= check("[R3] ... numeric value", p.key_rate_per_key_run(method="lp"), 0.106231, tol=1e-5)
ok &= check("[R4] rate per experimental run", p.key_rate_per_experimental_run(method="lp"), 0.053115, tol=1e-5)
ok &= check("[R5] LP min-entropy rate/key run", p.key_rate_per_key_run(method="lp", rate_type="min_entropy"), 0.028521, tol=1e-5)
g_sdp = float(p.eve_sdp_solver.eve_success_probability)
ok &= check("[R6] SDP (Moroder L1) P_guess bound", g_sdp, 0.5, tol=2e-3)
ok &= check("[R7] certified quantum rate/key run", p.key_rate_per_key_run(method="sdp"), 0.398852, tol=2e-3)

print("\n[R8] refined dual witness for y*=0:")
coeffs = p.eve_master_key_lp_solver.guess_bound_coeffs_by_y[0]
print(p.format_eve_guess_upper_bound_inequality_by_y(precision=4))
# Structure check: 1/8 on all (x, y=0, b); 1/4 on the four y=1 anti-correlated slots.
expected = np.zeros_like(coeffs)
expected[:, 0, :2] = 0.125
for x, b in [(0, 0), (1, 1), (2, 1), (3, 0)]:
    expected[x, 1, b] = 0.25
match = np.allclose(coeffs, expected, atol=1e-4)
print(f"  {'OK ' if match else 'FAIL'} witness coefficients match Eq. (witness) of the manuscript")
ok &= match
bound_at_data = float(np.sum(coeffs * np.asarray(p.scenario.data_numeric)))
ok &= check("[R8] witness value at data = P_guess", bound_at_data, g_lp, tol=1e-6)

print("\nALL CLAIMS VERIFIED" if ok else "\nSOME CLAIMS FAILED")
raise SystemExit(0 if ok else 1)

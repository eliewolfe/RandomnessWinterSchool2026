"""Verify every entry of the manuscript's operational-equivalence
restriction table (Sec. "Which operational equivalences power the key?",
Table III), and the surrounding Cabello claims.

For each protocol we compare Eve's exact operational-adversary guessing
probability (LP) under (a) the full learned set of measurement
operational equivalences and (b) only the completeness
(unit-measurement-trace) relations, via
ContextualityScenario.restricted_to_completeness_meas_opeqs().

Claims checked:
  [O1] equivalence counts (full -> completeness-only) per protocol;
  [O2] for every qubit protocol P_guess and the key rate are UNCHANGED;
  [O3] for the Cabello 18-ray set (automatically optimized where_key)
       P_guess rises 0.625 -> 0.671 and the reverse-Fano rate per key
       round falls 0.750 -> 0.658 (0.167 -> 0.146 per experimental run).

Runtime: ~3 minutes (LP only).
"""

from _protocols import (
    SINGLE_SETTING_WHERE_KEY,
    HEXAGON_ALIGNED_WHERE_KEY,
    aligned_bb84_scenario,
    cabello_18ray_scenario,
    check,
    hexagon_scenario,
    porac_scenario,
    protocol,
    rotated_bb84_scenario,
    xz_ring_scenario,
)

# name -> (builder, where_key, opeq counts (full, restricted), P_guess (full, restricted))
TABLE = {
    "aligned BB84":  (aligned_bb84_scenario, None,                     (1, 1), (0.750, 0.750)),
    "rotated BB84":  (rotated_bb84_scenario, SINGLE_SETTING_WHERE_KEY, (1, 1), (0.6464, 0.6464)),
    "(3,2)-PORAC":   (lambda: porac_scenario(1.0), None,               (2, 2), (0.7113, 0.7113)),
    "hexagon":       (hexagon_scenario, HEXAGON_ALIGNED_WHERE_KEY,     (3, 2), (0.750, 0.750)),
    "XZ ring (8,4)": (lambda: xz_ring_scenario(8, 4), None,            (5, 3), (0.625, 0.625)),
}

ok = True
for name, (build, wk, counts, guesses) in TABLE.items():
    sc_full = build()
    sc_rest = sc_full.restricted_to_completeness_meas_opeqs()
    print(f"{name}:")
    ok &= check("  [O1] opeq count, full", sc_full.opeq_meas_numeric.shape[0], counts[0], tol=0.1)
    ok &= check("  [O1] opeq count, restricted", sc_rest.opeq_meas_numeric.shape[0], counts[1], tol=0.1)
    p_full = protocol(sc_full, where_key=wk)
    p_rest = protocol(sc_rest, where_key=wk)
    g_full = p_full.eve_guess_master_key_average_y_lp
    g_rest = p_rest.eve_guess_master_key_average_y_lp
    ok &= check("  [O2] P_guess, full", g_full, guesses[0])
    ok &= check("  [O2] P_guess, restricted", g_rest, guesses[1])
    ok &= check("  [O2] rate unchanged (full = restricted)",
                p_rest.key_rate_per_key_run(method="lp"), p_full.key_rate_per_key_run(method="lp"), tol=1e-6)

print("Cabello 18-ray (automatically optimized where_key):")
sc = cabello_18ray_scenario()
sc_rest = sc.restricted_to_completeness_meas_opeqs()
ok &= check("  [O1] opeq count, full", sc.opeq_meas_numeric.shape[0], 26, tol=0.1)
ok &= check("  [O1] opeq count, restricted", sc_rest.opeq_meas_numeric.shape[0], 8, tol=0.1)
p_full = protocol(sc, where_key="Automatic")
p_rest = protocol(sc_rest, where_key="Automatic")
ok &= check("  [O3] P_guess, full", p_full.eve_guess_master_key_average_y_lp, 0.625)
ok &= check("  [O3] P_guess, restricted", p_rest.eve_guess_master_key_average_y_lp, 0.6711)
ok &= check("  [O3] rate/key run, full", p_full.key_rate_per_key_run(method="lp"), 0.750)
ok &= check("  [O3] rate/key run, restricted", p_rest.key_rate_per_key_run(method="lp"), 0.6579)
ok &= check("  [O3] rate/exp run, full", p_full.key_rate_per_experimental_run(method="lp"), 0.1667)
ok &= check("  [O3] rate/exp run, restricted", p_rest.key_rate_per_experimental_run(method="lp"), 0.1462)

print("\nALL CLAIMS VERIFIED" if ok else "\nSOME CLAIMS FAILED")
raise SystemExit(0 if ok else 1)

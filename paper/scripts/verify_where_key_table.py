"""Verify every entry of the manuscript's where_key design table
(Sec. "Designing the key map", Table II) against the operational
adversary (LP, exact; reverse-Fano; Alice holding the master key).

For each protocol the script computes the key rate per key-generating
round and per experimental run under (a) the trivial where_key (every x
for every y) and (b) the automatically optimized where_key, and checks
them against the table. It also checks the optimizer's chosen
assignments highlighted in the text (hexagon: each setting keys on its
own antipodal pair; Cabello: each context keys on its own four rays).

Runtime: ~2 minutes (LP only, no SDP needed).
"""

from _protocols import (
    SINGLE_SETTING_WHERE_KEY,
    cabello_18ray_scenario,
    check,
    hexagon_scenario,
    protocol,
    rotated_bb84_scenario,
    xz_ring_scenario,
)

# protocol -> (builder, trivial (key, run), optimized (key, run, p_key))
TABLE = {
    "XZ ring (8,4)":   (lambda: xz_ring_scenario(8, 4),   (0.1072, 0.1072), (0.2052, 0.1539, 0.750)),
    "XZ ring (12,6)":  (lambda: xz_ring_scenario(12, 6),  (0.1676, 0.1676), (0.4927, 0.2464, 0.500)),
    "XZ ring (16,8)":  (lambda: xz_ring_scenario(16, 8),  (0.2013, 0.2013), (0.4505, 0.2816, 0.625)),
    "hexagon (6,3)":   (hexagon_scenario,                 (0.0166, 0.0166), (0.5000, 0.1667, 1 / 3)),
    "Cabello 18-ray":  (cabello_18ray_scenario,           (-0.2733, -0.2733), (0.7500, 0.1667, 2 / 9)),
}

ok = True
for name, (build, trivial, optimized) in TABLE.items():
    sc = build()
    print(f"{name}:")
    p = protocol(sc, where_key=None)
    ok &= check("  trivial rate/key run", p.key_rate_per_key_run(method="lp"), trivial[0])
    ok &= check("  trivial rate/exp run", p.key_rate_per_experimental_run(method="lp"), trivial[1])
    p = protocol(sc, where_key="Automatic")
    ok &= check("  optimized rate/key run", p.key_rate_per_key_run(method="lp"), optimized[0])
    ok &= check("  optimized rate/exp run", p.key_rate_per_experimental_run(method="lp"), optimized[1])
    ok &= check("  optimized key fraction", p.key_generation_probability_per_run, optimized[2], tol=1e-6)
    if name == "hexagon (6,3)":
        aligned = tuple(tuple(row) for row in p.where_key) == ((0, 3), (1, 4), (2, 5))
        print(f"  {'OK ' if aligned else 'FAIL'} optimizer picks the aligned antipodal pairs")
        ok &= aligned
    if name == "Cabello 18-ray":
        own_rays = all(len(row) == 4 for row in p.where_key)
        print(f"  {'OK ' if own_rays else 'FAIL'} optimizer keys each context on 4 preparations")
        ok &= own_rays

print("rotated BB84 (single setting):")
p = protocol(rotated_bb84_scenario(), where_key=SINGLE_SETTING_WHERE_KEY)
pa = protocol(rotated_bb84_scenario(), where_key="Automatic")
ok &= check("  single-setting rate/key run", p.key_rate_per_key_run(method="lp"), 0.1062)
ok &= check("  optimizer leaves symmetric protocol trivial (rate/exp run)",
            pa.key_rate_per_experimental_run(method="lp"), 0.1062)
ok &= check("  ... with key fraction", pa.key_generation_probability_per_run, 1.0, tol=1e-9)

print("\nALL CLAIMS VERIFIED" if ok else "\nSOME CLAIMS FAILED")
raise SystemExit(0 if ok else 1)

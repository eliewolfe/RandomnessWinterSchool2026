# Verification scripts for the manuscript

Every numerical claim in `paper/main.tex` is substantiated by one of the
scripts below. Each `verify_*` script recomputes the claimed quantities from
scratch, compares them to the values printed in the manuscript (`OK`/`FAIL`
per claim), and exits nonzero on any mismatch. Each `scan_*` script does the
same for a figure's parameter sweep and writes the figure's data file under
`paper/data/`. The `fig_*` scripts render `paper/figures/*.pdf` from those
data files.

Run from this directory (`uv run python <script>` from the repo root works
too). MOSEK (with license) is required for the SDP parts; the LP parts run on
HiGHS. Unless stated otherwise, quantum-adversary bounds use
`npa_level_bob = npa_level_eve = 1`, whose moment matrix is indexed by the
words {1, Bob ops, Eve ops, Bob·Eve products} — level 1 of the Moroder-type
hierarchy ("NPA 1+ABE"), with Alice's preparations indexing the blocks.
Where a script uses level 2 it says so and why.

| Manuscript location | Script | Runtime |
|---|---|---|
| Sec. Results A, Table I, Eq. (witness) | `verify_single_setting_rates.py` | ~1 min |
| Sec. Results A, Fig. 1, thresholds, closed forms | `scan_single_setting_visibility.py` | ~20 min (`--lp-only`: seconds) |
| Sec. Results B, noiseless PORAC numbers | `verify_porac_separation.py` | ~1 min |
| Sec. Results B, Fig. 2, threshold, closed forms | `scan_porac_visibility.py` | ~10 min |
| Sec. Results C, Table II | `verify_where_key_table.py` | ~2 min |
| Sec. Results D, Table III | `verify_opeq_restriction_table.py` | ~3 min |
| Sec. Results E, Fig. 3, closed forms | `verify_witness_only_rates.py` | ~6 min |
| Sec. Results F, QRAC family table | `scan_qrac_family.py` | ~15 min |
| Fig. 1 rendering | `fig_single_setting.py` | seconds |
| Fig. 2 rendering | `fig_porac.py` | seconds |
| Fig. 3 rendering | `fig_witness_only.py` | seconds |

Shared protocol definitions live in `_protocols.py` (one definition per
protocol, imported by every script), including the (n,d) MUB-QRAC and
d-ary oblivious-multiplexing constructions (optimal states, obliviousness
see-saw, exact-projection and POVM-repair helpers).

Performance note: the LP backend here is MOSEK interior point. The RAC
scenarios have massively degenerate optimal faces on which simplex methods
(HiGHS, MOSEK simplex) stall by orders of magnitude, while interior point
solves in about a second; all optima agree. Operational-equivalence bases
are discovered by NumPy SVD (orthonormal, well conditioned) -- see
`contextualityqkd.linalg_utils.null_space_basis` -- and can be lifted to
exact symbolic coefficients with `lift_nullspace_to_symbolic`.

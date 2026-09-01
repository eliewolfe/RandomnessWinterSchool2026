# Results-section plan

Detailed working plan for Section "Results / Case studies" of the manuscript
(`paper/main.tex`, currently `\section{Case studies}`, to be renamed
`\section{Results}` with the subsections below). Each result item R1–R6 lists:
purpose, protocol choice, what the existing code already supports, new code
required, the exact computations to run, the deliverables for the paper
(text, tables, figures), and risks/open questions. A phased execution plan
with dependencies follows at the end.

Conventions: "OPT adversary" = eavesdropper constrained only by the
operational theory (our LP, which is **exact/tight** for this adversary);
"quantum adversary" = eavesdropper constrained by quantum theory (our SDP,
which — being a level-limited relaxation in an NPA-style approximation
hierarchy — only **upper-bounds** Eve's guessing probability, hence
**lower-bounds** the certified key rate; tightness improves with hierarchy
level). This LP-tight vs. SDP-one-sided asymmetry must be stated wherever
LP and SDP numbers appear side by side, and prominently in R2.

---

## R1. Pedagogical warm-up: key from a single measurement setting of Bob

**Purpose.** Present a complete key-rate analysis *before* introducing Eve's
quantum memory and individual adaptive (setting-dependent, delayed) attacks.
If only one setting y* of Bob is ever key-generating, then on key rounds Eve
gains nothing from delaying her probe measurement: her optimal attack is a
single fixed measurement chosen in advance, so no memory and no adaptivity
are needed to analyze it. This lets the manuscript introduce the LP/SDP
machinery on the simplest adversary and *then* motivate the general paradigm
(memory + setting-dependent product measurements) as the new ingredient
required once several y's generate key.

**Protocol choice.** BB84-as-contextuality-scenario (`demos/qkd_BB84.py`)
with `where_key = [[0, 1], []]` — key established only from Bob's
computational-basis setting; the second setting is used purely for parameter
estimation (witnessing contextuality). Alternative if the numbers are more
striking: the XZ-ring (`qkd_xz_ring.py`) restricted to one y.

**Existing code support.** `where_key` already accepts per-y lists;
`eve_lp` returns per-y guessing probabilities and the protocol layer
aggregates. **To verify:** empty rows `where_key[y] = []` propagate cleanly
through (a) `eve_lp` (empty objective → the per-y LP is skipped; the code
currently leaves `NaN` — confirm the aggregation into
`key_rate_per_experimental_run` treats such y as non-key rather than
poisoning averages), (b) the SDP pathway, and (c) all `print_*`/`format_*`
reporting. Likely a small patch + a regression test (`tests/`): "empty
where_key row is handled identically in LP and SDP reporting."

**Computations.**
1. Key rate (min-entropy and reverse-Fano) vs. depolarizing noise for the
   single-key-setting BB84 protocol, OPT adversary (LP).
2. Same against quantum adversary (SDP, NPA level 1; level 2 if tractable).
3. The per-y dual witness inequality for y*, printed and simplified — this
   is the first "witness" the reader meets.

**Paper deliverables.** Subsection VI.A. One table (noiseless rates, both
uncertainty measures, both adversaries), one figure (rate vs. noise), one
displayed witness inequality. Text: the pedagogical framing above, plus the
explicit remark that from VI.B onward Eve keeps her probe until y is
announced.

**Risks.** Low. Only the empty-row plumbing.

---

## R2. Quantum vs. OPT adversary gap: the (3,2)-PORAC

**Purpose.** Exhibit a protocol whose key rate is **zero against the OPT
adversary but provably positive against the quantum adversary** — the
headline separation between the two security models, and the payoff of the
SDP machinery.

**Protocol.** The (3,2)-parity-oblivious random access code
(`demos/qkd_porac_3_2.py`, `build_porac_scenario(eta)`), with the
parity-obliviousness conditions as the preparation operational
equivalences.

**Status of the claim (to be reconfirmed by fresh runs).** Per prior runs:
our LP gives Eve guessing probability high enough that the OPT key rate is
zero; our SDP (Naimark-unitary nonprojective pathway, U-only generators,
NPA level 1) certifies a positive rate. External anchor: Chaturvedi, Farkas
& Wright, *Quantum* **5**, 484 (2021), arXiv:2010.05853 [`ChaturvediFarkasWright2021`
in `references.bib`] present a semi-device-independent QKD scheme powered by
the quantum advantage of parity-oblivious random access codes and bound the
quantum adversary via their SDP hierarchy for contextuality scenarios. We
compare our certified rates to theirs (same scenario normalization must be
checked carefully — their figure of merit and sifting conventions may
differ; a conversion appendix may be needed).

**Existing code support.** Complete: LP + SDP paths, eta-scan scaffolding in
the demo, per-y and averaged dual witnesses.

**New code.** None required; optionally a small script
`paper/scripts/porac_scan.py` producing the table/figure data
(rates vs. eta at NPA levels 1 and, resources permitting, 2).

**Computations.**
1. Noiseless (eta = 1): LP guessing probability (exact) → OPT key rate 0;
   SDP guessing bound at levels 1, 2 → certified quantum key rate.
2. Rate vs. eta curve for the quantum adversary; the eta threshold below
   which our certificate degenerates.
3. Cross-check against the rates reported in arXiv:2010.05853.

**Paper deliverables.** Subsection VI.B. Table: guessing probability and key
rate, {OPT-LP exact, quantum-SDP level 1, quantum-SDP level 2,
Chaturvedi–Farkas–Wright}. Figure: rate vs. eta. **Mandatory text:** (i) the
LP is tight — zero OPT rate is a *theorem about the protocol*, not a failure
of our bound; (ii) the SDP is a relaxation within an approximation
hierarchy, so it upper-bounds Eve's guessing probability and thereby
certifies (lower-bounds) the quantum key rate; the true quantum rate may be
higher; hierarchy levels quantify the gap.

**Risks.** Level-2 SDP size; normalization mismatch with the external paper.

---

## R3. Key-rate improvement from nontrivial `where_key`

**Purpose.** Show a protocol where restricting which preparations x are
key-generating for each y strictly **increases the key rate per experimental
run** — demonstrating that "key rate" for prepare-and-measure contextuality
protocols is a protocol-design quantity, not just a property of the
behavior, and showcasing the automatic `where_key` optimizer.

**Protocol candidates (scan, then pick the cleanest).** In order of
expected payoff: the XZ-ring family (`qkd_xz_ring.py` — many preparations,
only some well-correlated with each y), the hexagon
(`qkd_hexagon_projective.py`), the 18-ray Cabello and 24-ray Peres KS sets
(`qkd_cabello_18ray.py`, `qkd_peres_24ray.py`). The improvement mechanism:
dropping poorly-correlated x's lowers the error-correction cost per key dit
faster than it shrinks the fraction of key rounds.

**Existing code support.** Complete: `where_key="Automatic"` optimizer with
clustering tolerance / tie-break knobs, plus
`print_where_key_optimization_best_stage`. Both `key_rate_per_key_run` and
`key_rate_per_experimental_run` are exposed, which is exactly the
distinction this subsection dramatizes.

**New code.** None; a comparison script `paper/scripts/where_key_scan.py`
iterating over the demo scenarios and reporting (rate with trivial
`where_key`) vs. (rate with optimized `where_key`), per run type.

**Computations.** For each candidate: trivial vs. optimized `where_key`,
key rate per key run AND per experimental run, LP adversary (SDP for the
winner only, as confirmation).

**Paper deliverables.** Subsection VI.C. Table: per-protocol trivial vs.
optimized rates (both bookkeepings). Text: formal definitions of the two
rates; the trade-off; a short description of the optimizer's search space
(subsets of x per y, clustering of equivalent stages).

**Risks.** Possible that for all current demos the optimum is trivial; if
so, design a deliberately lopsided variant (e.g., an XZ-ring with unevenly
spaced preparations) — cheap to generate with the existing quantum-scenario
constructors.

---

## R4. Which operational equivalences power the key? Completeness-only vs. all

**Purpose.** Isolate the role of measurement-side operational equivalences
*beyond* the trivial ones. The completeness relations (all outcomes of a
given setting sum to the unit effect — "unique measurement trace") are the
prepare-and-measure analogue of no-signalling in Bell-based QKD: in NS-QKD
the eavesdropper's ignorance flows solely from no-signalling. Our question:
how much *additional* constraint on Eve flows from the nontrivial identities
among Bob's effects that quantum mechanics supplies (e.g., the same effect
appearing in two settings, KS-style orthogonality relations)? This is the
conceptual bridge between our framework and nonlocality-based key
distribution, and deserves its own subsection.

**New code (required).** Currently `ContextualityScenario` learns ALL
measurement operational equivalences by linear algebra from the given
effects. Add a scenario-level restriction knob so both solvers inherit it
automatically:
- `ContextualityScenario(..., meas_opeq_policy=...)` with values
  `"all"` (default, current behavior) and `"completeness_only"`; possibly
  also `"custom"` with an explicit list.
- Under `"completeness_only"`, synthesize exactly the |Y| completeness
  opeqs directly (coefficient tensor d[y,b] = δ_{y,y0} minus the same for a
  reference setting, or equivalently "sum_b e_{b|y} equals sum_b e_{b|y'}"
  written in the code's homogeneous opeq format — decide the exact encoding
  to match `opeq_meas_numeric` conventions) rather than filtering the
  learned null-space basis (filtering is fragile: the learned basis need
  not contain the completeness relations as distinguished elements).
- Everything downstream (`eve_lp`, `eve_sdp`, contextual-fraction module,
  printing) consumes `scenario.opeq_meas_numeric`, so no solver changes
  should be needed. Add tests: (a) completeness-only is a relaxation —
  Eve's LP guessing probability is ≥ the all-opeqs value on every demo;
  (b) for a scenario whose only measurement opeqs ARE completeness, the two
  policies agree.
- Preparation-side opeqs stay untouched (full).

**Computations.** For every demo protocol: Eve's guessing probability and
key rate under `"completeness_only"` vs. `"all"`, LP first, SDP for the
interesting cases. Classify protocols into: (i) key survives on
completeness alone (NS-like security); (ii) key exists only thanks to
further effect identities; (iii) no key either way.

**Paper deliverables.** Subsection VI.D. Table of the classification;
discussion contrasting with Bell/NS-QKD (cite standard NS-QKD literature —
Barrett–Hardy–Kent and follow-ups; to be added to the bib); a dual-witness
pair for one protocol showing how the witness changes when extra identities
are switched on.

**Risks.** Encoding of "completeness" in the homogeneous opeq format needs
care (the code's opeq convention is sums-equal-zero between contexts;
confirm how unit-effect normalization is represented). Medium code effort,
mostly in `scenario.py` + tests.

---## R5 (OPTIONAL — final phase, on request only). Comparison against dimension-bound-only security

**Purpose.** Prior semi-device-independent work — notably Pawłowski &
Brunner, *Phys. Rev. A* **84**, 010302(R) (2011), arXiv:1103.4105
[`PawlowskiBrunner2011`] — obtains key rates for protocols very close to
ours from a **dimension bound alone** (Hilbert-space dimension ≤ 2),
certified through dimension witnesses / QRAC values. Our operational
identities effectively bound the system's dimension too, so the comparison
suggests itself: for the same observed behavior, does the
operational-equivalence assumption certify **more** key than the dimension
assumption?

**Fairness caveat (must be discussed in the text, whatever the numbers).**
The two assumption sets are incomparable in general: a dimension bound is a
device-level physical assumption; operational equivalences are promises
about the source/measurement statistics that are experimentally enforceable
by post-processing (see R6). Neither implies the other. The honest framing:
same experimental data, two different trust models, report both rates.

**New code (substantial — why this is optional).** A dimension-constrained
Eve SDP: drop the operational-equivalence constraints and instead constrain
the message system to dimension d (qubit). Options, in increasing effort:
(a) reuse published analytic rates from arXiv:1103.4105 for the BB84/QRAC
protocol and only compute our side; (b) see-saw lower bounds + NPA-with-
dimension (e.g., NV hierarchy) upper bounds for the dimension-constrained
guessing probability. Recommendation: (a) for the paper, (b) only if a
referee asks.

**Deliverables.** Subsection VI.E (or an appendix): one table comparing our
key rate vs. the dimension-bound rate on the shared protocol; the fairness
discussion.

---

## R6. Key rates from noncontextuality-inequality violation alone

**Purpose & verdict on "does it make sense".** Yes — and it is the natural
prepare-and-measure analogue of DIQKD's "key rate as a function of CHSH
violation". Constraining Eve by (i) the operational equivalences and (ii)
only a lower bound w ≥ w_obs on one linear witness (instead of full
consistency with P(b|x,y)) is a *relaxation* of the current LP/SDP, so the
resulting rate is a lower bound on the full-data rate and inherits its
security proof unchanged. The scientific questions are: does any witness
threshold give a positive rate, and how much rate is lost relative to
full-data conditioning? Expect a DIQKD-like picture: a threshold violation
below which the rate is zero, and a monotone curve above it. There is one
structural subtlety absent from DIQKD that the text must address: the
guessing objective refers to specific settings (x, y) and the key map
k(x,y), so the objective itself is still built from the full scenario
structure — only the *data-consistency constraints* are being coarse-grained
to a single scalar. Normalization constraints sum_{b,e} P(b,e|x,y) = 1 must
be retained (they are implied by data consistency today, and would otherwise
be lost).

**Where the witness comes from (existing code).** Two natural sources, both
already extractable: the dual of the contextual-fraction LP
(`NoncontextualityAssessment.inequality` / `inequality_bound`, with
L2-refined coefficients), and the per-y Eve-guessing dual witnesses
(`eve_guess_master_key_upper_bound_coeffs`). Use the contextual-fraction
witness for the headline (it is the "noncontextuality inequality" a
experimentalist would quote), and mention the guessing-dual witness as the
self-consistent optimum.

**New code (small, both solvers).** In `eve_lp.QKDNoncontextualLP` and
`eve_sdp.QKDNoncontextualSDP`, add a constructor mode
`data_constraint="full"` (default) vs.
`data_constraint=("witness", coeffs, bound)`:
- full: current per-(x,y,b) equalities with observed data;
- witness: per-(x,y) normalization equalities + the single inequality
  sum_{x,y,b} c[x,y,b] * sum_e P(b,e|x,y) ≥ bound (LP) and its moment-matrix
  counterpart (SDP). Dual reporting should still work (one named dual for
  the witness row).
Protocol layer: a thin wrapper `ContextualityProtocol.key_rate_from_witness
(coeffs, bound, ...)` plus a scan utility producing rate-vs-violation
curves. Tests: witness-mode rate ≤ full-data rate on every demo; witness
mode with the *complete list* of facet equalities reproduces full-data rate.

**Computations.**
1. For the R1 protocol and the (3,2)-PORAC: rate vs. witness-violation
   curves (LP/OPT and SDP/quantum), marking the observed violation.
2. Identify at least one protocol with positive rate purely from the
   violation — the "experimentally friendly" headline.

**Paper deliverables.** Subsection VI.F. Figure: key rate vs. witness value
(the PM analogue of rate-vs-CHSH). Text: the relaxation argument; the
experimental-friendliness point — ideal operational equivalences tied to a
given noncontextuality inequality can always be enforced in the lab by
Spekkens-style "secondary procedures" post-processing (lapidation), as
opposed to cut-to-your-cloth analyses; cite Mazurek et al., *Nat. Commun.*
**7**, 11780 (2016) [`Mazurek2016`] for secondary procedures and Zhang et
al., arXiv:2507.01122 [`Zhang2025lapidation`] for the
lapidation-vs-cut-to-your-cloth contrast. **ACTION: Yujie to confirm
arXiv:2507.01122 is the intended lapidation reference** (web search could
not verify the term's occurrence in the text; if it lives in another
Spekkens–Zhang manuscript, swap the entry).

**Risks.** Low-medium. The SDP witness mode needs the witness expressed in
the SDP's moment variables — straightforward since data constraints already
are. Degenerate duals may make the rate-vs-violation curve piecewise-linear
with plateaus; that is fine (and interesting) but should be anticipated.

---

## Proposed manuscript layout for the Results section

- VI. Results
  - VI.A Warm-up: single-setting key generation (R1)
  - VI.B Separating quantum from operational adversaries: the (3,2)-PORAC (R2)
  - VI.C Designing the key map: nontrivial `where_key` (R3)
  - VI.D Which operational equivalences power the key? (R4)
  - VI.E Key from a noncontextuality-inequality violation alone (R6)
  - [VI.F Comparison with dimension-bounded security (R5) — optional]
- Ordering rationale: pedagogy (A) → headline separation (B) → protocol
  design (C) → conceptual analysis (D) → experimental friendliness (E).
  R5, if commissioned, slots after D or into an appendix.

## Figures & tables inventory

| Item | Type | Result |
|---|---|---|
| T1 | Table: single-setting BB84 rates (LP/SDP × min-entropy/reverse-Fano) | R1 |
| F1 | Figure: R1 rate vs. noise | R1 |
| T2 | Table: PORAC guessing prob & rate — LP exact, SDP L1/L2, CFW2021 | R2 |
| F2 | Figure: PORAC quantum rate vs. eta | R2 |
| T3 | Table: trivial vs. optimized where_key, both rate bookkeepings | R3 |
| T4 | Table: completeness-only vs. all measurement opeqs, per protocol | R4 |
| F3 | Figure: key rate vs. witness violation (PM analogue of rate-vs-CHSH) | R6 |
| W1–W3 | Displayed witness inequalities (R1, R4 contrast pair) | R1, R4 |

## Code work items

| ID | Item | Module(s) | Size | Blocks |
|---|---|---|---|---|
| C1 | Empty `where_key` rows end-to-end + regression test | eve_lp, eve_sdp, protocol | S | R1 |
| C2 | `meas_opeq_policy="completeness_only"` at scenario level + tests | scenario | M | R4 |
| C3 | `data_constraint="witness"` mode in LP and SDP + protocol wrapper + tests | eve_lp, eve_sdp, protocol | M | R6 |
| C4 | Scan scripts under `paper/scripts/` (porac_scan, where_key_scan, witness_curve) | new | S | R2, R3, R6 |
| C5 | (optional) dimension-constrained Eve SDP | new module | L | R5 |

## Phased execution

- **Phase 0 (infrastructure).** TeX toolchain in the session (done — being
  reinstalled after a mirror hiccup); `latexmk` build of `main.tex`;
  PDF delivered on every substantive paper edit.
- **Phase 1.** C1 → run R1 computations → draft VI.A.
- **Phase 2.** R2 computations (no code) + C4 scan scripts → draft VI.B,
  including the LP-tight vs. SDP-hierarchy framing and the CFW2021
  cross-check.
- **Phase 3.** C4 scan → pick R3 winner (or synthesize a lopsided ring) →
  draft VI.C.
- **Phase 4.** C2 → R4 sweep over all demos → classification table → draft
  VI.D.
- **Phase 5.** C3 → R6 curves → draft VI.E; confirm the lapidation citation
  with Yujie.
- **Phase 6 (optional, on explicit request).** C5 or literature-value route
  → R5 → draft VI.F/appendix.
- Dependencies: phases 1–3 are mutually independent (any order); 4 and 5
  are independent of each other but both benefit from 1–2 being settled
  (shared table formats); 6 last.

## Reference bookkeeping (added to references.bib now)

- `ChaturvediFarkasWright2021` — A. Chaturvedi, M. Farkas, V. J. Wright,
  "Characterising and bounding the set of quantum behaviours in
  contextuality scenarios", Quantum 5, 484 (2021), arXiv:2010.05853.
  (The "Vicky et al." anchor for R2: SDI-QKD powered by PORAC quantum
  advantage, quantum adversary bounded by an SDP hierarchy.)
- `PawlowskiBrunner2011` — M. Pawłowski, N. Brunner, "Semi-device-independent
  security of one-way quantum key distribution", PRA 84, 010302(R) (2011),
  arXiv:1103.4105. (R5 anchor.)
- `Mazurek2016` — M. D. Mazurek, M. F. Pusey, R. Kunjwal, K. J. Resch,
  R. W. Spekkens, "An experimental test of noncontextuality without
  unphysical idealizations", Nat. Commun. 7, 11780 (2016). (Secondary
  procedures, R6.)
- `Zhang2025lapidation` — Y. Zhang et al. (with R. W. Spekkens),
  arXiv:2507.01122. (Lapidation vs. cut-to-your-cloth — **pending Yujie's
  confirmation**.)
- Still to add when drafting VI.D: NS-QKD references (Barrett–Hardy–Kent
  et al.).

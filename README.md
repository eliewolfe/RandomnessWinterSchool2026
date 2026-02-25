# contextualityqkd: Bob-Outcome QKD from Contextuality

This repository studies quantum key distribution (QKD) protocols where:

- Alice chooses a preparation setting `x`.
- Bob chooses a measurement setting `y` and gets outcome `b`.
- Security is analyzed against Eve's best possible guess of Bob's outcome, constrained only by observed operational equivalences and nonnegativity.

The project is designed around a clean separation between:

1. **Scenario modeling** (`ContextualityScenario`): represent and validate `p(b|x,y)` and contextuality constraints.
2. **Protocol analysis** (`ContextualityProtocol`): define key-generating runs (`where_key`) and compute Alice/Eve guessing, uncertainties, and key-rate metrics.

## Why This Design

Earlier designs mixed physical modeling and protocol-specific guessing tasks in one class. The current architecture intentionally separates concerns:

- `ContextualityScenario` answers: "What behavior and operational constraints does this experiment define?"
- `ContextualityProtocol` answers: "Given those constraints, what key can be extracted under a specific postselection/key-generation rule?"

This separation makes it easier to:

- Reuse one scenario with multiple protocol choices (`where_key`).
- Compare protocol tradeoffs without rebuilding the physical model.
- Extend Eve analysis methods in the future (LP now, SDP later).

## Conceptual Pipeline

For all demos and most user workflows:

1. Build a physical model (GPT or quantum states/effects).
2. Construct a `ContextualityScenario` to obtain `p(b|x,y)` and OPEQs.
3. Define protocol key rule `where_key[y]` (which `x` are key-eligible after Bob reveals `y`).
4. Construct `ContextualityProtocol(scenario, where_key=...)`.
5. Read/print Alice metrics, Eve LP bounds, and key rates.
6. Optionally compute contextuality metrics (`dephasing robustness`, `contextual fraction`).

## Mathematical Objects

### Scenario data

The core behavior is Bob-only:

- `data[x, y, b] = p(b|x,y)`
- Shape is padded to `(X, Y, B_max)`.
- If a measurement has fewer outcomes than `B_max`, padded entries are structurally zero.

`ContextualityScenario` stores:

- `X_cardinality`, `Y_cardinality`, `B_cardinality` (`B_max`)
- `b_cardinality_per_y`
- `valid_b_mask`
- preparation OPEQs with shape `(N_prep, X)`
- measurement OPEQs with shape `(N_meas, Y, B_max)`

### Operational equivalences (OPEQs)

When not provided, OPEQs are discovered from null spaces of the data table:

- Preparation OPEQs satisfy `sum_x c[x] p(b|x,y) = 0` for all `(y,b)`.
- Measurement OPEQs satisfy `sum_{y,b} d[y,b] p(b|x,y) = 0` for all `x`.

These constraints are what limit Eve's compatible decomposition models.

### Protocol key rule: `where_key`

`where_key` is a list/array with one row per `y`:

- `where_key[y]` = set of `x` values treated as key-eligible when Bob announces `y`.
- If `where_key=None`, protocol defaults to all `x` for all `y`.
- Rows are canonicalized (dedup + sort).
- Empty rows are allowed.

This supports postselection-style protocols naturally.

## Under the Hood

### 1) Scenario layer (`contextualityqkd/scenario.py`)

`ContextualityScenario` does all data hygiene and structural validation:

- accepts dense or ragged behavior input,
- enforces per-`(x,y)` normalization,
- enforces structural zeros on padded outcomes,
- discovers or validates OPEQs,
- exposes formatted symbolic/numeric printing helpers.

It also wraps contextuality LP routines:

- `compute_dephasing_robustness(...)`
- `compute_contextual_fraction(...)`
- `compute_noncontextual_fraction(...)`

### 2) Contextuality LP layer (`contextualityqkd/contextuality.py`)

This layer builds assignment-ray cones and solves LPs for:

- **dephasing robustness**: minimal `r` such that `(1-r)P + rD` is noncontextual,
- **contextual fraction** and **noncontextual fraction**.

The default dephasing target is `x`-independent at fixed `y`:

- `D(b|x,y) = mean_x p(b|x,y)`.

### 3) Eve LP layer (`contextualityqkd/randomness.py`)

Eve's Bob-guessing LP is solved per `y` with hotstart reuse:

- objective for each `y`: maximize average success over `x in where_key[y]`,
- constraints enforce compatibility with observed `p(b|x,y)` and all OPEQs,
- output is `P_E^guess(B|y,key)` for each `y`.

Entropy helper functions live here:

- `min_entropy(p_guess) = -log2(p_guess)`
- `reverse_fano_bound(p_guess)` (lower bound on Shannon entropy)
- `binary_entropy(p)`

### 4) Protocol layer (`contextualityqkd/protocol.py`)

`ContextualityProtocol` is cache-first (`cached_property`) for deterministic no-arg outputs.

It provides:

- key-eligibility bookkeeping (`key_mask_xy`, `key_generation_probability_per_run`, ...),
- Alice metrics (`guess` and `uncertainty`) both by `(x,y)` and averaged forms,
- Eve LP metrics (`*_lp` naming convention),
- key rates (reverse-Fano and min-entropy variants),
- native protocol print/format methods for report-ready outputs.

Alice `(x,y)` reports are masked to key-eligible entries (`--` for non-key pairs).

## Key Metrics and Their Meaning

Let `K_y = where_key[y]`.

### Alice

- `alice_guess_bob_by_xy[x,y] = max_b p(b|x,y)`
- `alice_guess_bob_by_y_key[y] = average_{x in K_y} max_b p(b|x,y)`
- `alice_uncertainty_bob_by_xy[x,y] = H(B|x,y)`
- `alice_uncertainty_bob_by_y_key[y] = average_{x in K_y} H(B|x,y)`

Weighted scalars average over key-eligible pairs `(x,y)`.

### Eve (LP)

For each `y`, Eve optimizes a compatible model to maximize Bob-guessing success averaged over `x in K_y`:

- `eve_guess_bob_by_y_lp[y]`
- `eve_guess_bob_average_y_lp` (uniform over non-empty `y`)

Uncertainty lower bounds derived from Eve's guess probability:

- `eve_uncertainty_bob_min_entropy_by_y_lp`
- `eve_uncertainty_bob_reverse_fano_by_y_lp`

### Key rate

Per-`y` (reverse-Fano primary in demos):

- `r_y = H_E^lb(B|y,key) - H_A(B|X,y,key)`

Then:

- **bits per key-generating run**: weighted average of `r_y` by `|K_y|`
- **bits per experimental run**:
  `key_generation_probability_per_run * bits_per_key_generating_run`

If no key-eligible pairs exist, per-experimental-run rate is defined as `0.0`.

## Environment and Running

Per project instructions, use conda env `py13`:

```powershell
C:\Users\elupu\miniconda3\Scripts\conda.exe run -n py13 python -m contextualityqkd.demos.qkd_porac_3_2
```

Run all kept demos:

```powershell
C:\Users\elupu\miniconda3\Scripts\conda.exe run -n py13 python -m contextualityqkd.demos.run_all
```

Run tests:

```powershell
C:\Users\elupu\miniconda3\Scripts\conda.exe run -n py13 python -m unittest discover -s tests -v
```

## Minimal Usage Example

```python
import numpy as np
from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.quantum import QuantumContextualityScenario

# Build or load quantum states/effects
scenario = QuantumContextualityScenario(
    quantum_states=np.asarray(..., dtype=complex),
    quantum_effects=np.asarray(..., dtype=complex),
)

# Optional postselection; None means all x for all y
where_key = None
protocol = ContextualityProtocol(scenario, where_key=where_key)

# Print protocol report blocks
protocol.print_alice_guessing_metrics()
protocol.print_alice_uncertainty_metrics()
protocol.print_eve_guessing_metrics_lp()
protocol.print_eve_uncertainty_metrics_reverse_fano_lp()
protocol.print_key_rate_summary_reverse_fano_lp()

# Contextuality diagnostics
scenario.print_contextuality_measures()
```

## Demo Guide

This repository keeps four QKD demos, all built around `ContextualityProtocol`.

### 1) `qkd_hexagon_projective.py`

- **System**: qubit-like hexagon construction using GPT vectors from six projectors on the `xz` plane.
- **Preparations**: 6 (`x = 0..5`).
- **Measurements**: 3 binary projective settings (`y = 0..2`).
- **Key rule**: `where_key[y]` equals the two preparation indices in that measurement basis.
- **What it teaches**:
  - simplest nontrivial postselection protocol,
  - Alice has deterministic knowledge on key-eligible pairs,
  - Eve's LP bound and key-rate accounting are easy to inspect.

### 2) `qkd_cabello_18ray.py`

- **System**: Cabello 18-ray Kochen-Specker set.
- **Preparations**: 18 rays.
- **Measurements**: 9 contexts, each with 4 outcomes.
- **Key rule**: for each `y`, key-eligible `x` are exactly the rays in that context.
- **What it teaches**:
  - larger KS contextuality instance,
  - nontrivial OPEQ structure discovered from data,
  - protocol-level masking of `(x,y)` metrics becomes important for readability.

### 3) `qkd_peres_24ray.py`

- **System**: Peres 24 rays grouped into 6 disjoint 4-ray bases.
- **Preparations**: 24.
- **Measurements**: 6 bases, 4 outcomes each.
- **Key rule**: `where_key[y]` = the 4 rays in basis `y`.
- **What it teaches**:
  - clean disjoint-basis postselection,
  - high per-key-run rate can coexist with low key-generation probability,
  - why "bits per key run" and "bits per experimental run" are both needed.

### 4) `qkd_porac_3_2.py`

- **System**: `(3,2)`-PORAC qubit construction.
- **Preparations**: 8 (three input bits encoded into one index `x`).
- **Measurements**: 3 Pauli-based binary measurements.
- **Key rule**: default `where_key=None` (all `x` for all `y`).
- **What it teaches**:
  - no postselection case (`key_generation_probability_per_run = 1`),
  - explicit comparison of discovered prep OPEQ subspace with article constraints,
  - useful stress test for LP-derived Eve bounds.

## Interpreting Common Patterns Across Demos

A recurring pattern is that per-experimental-run key can be limited by:

1. small key-generation probability (strong postselection), even if per-key-run rate is high, or
2. high Eve guessability when no postselection is used.

This is expected: improving reliability for Alice by shrinking `where_key` often lowers throughput.
The protocol layer exposes both quantities so this tradeoff is explicit.

## API Surface (Most Used)

Imports:

```python
from contextualityqkd.scenario import ContextualityScenario
from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.quantum import GPTContextualityScenario, QuantumContextualityScenario
```

Common scenario methods:

- `print_probabilities(...)`
- `print_operational_equivalences(...)`
- `compute_dephasing_robustness(...)`
- `compute_contextual_fraction(...)`

Common protocol properties:

- Alice: `alice_guess_bob_by_xy`, `alice_uncertainty_bob_by_xy`, key-conditioned variants
- Eve LP: `eve_guess_bob_by_y_lp`, reverse-Fano/min-entropy uncertainty variants
- Key rates: `key_rate_per_key_run_reverse_fano_lp`, `key_rate_per_experimental_run_reverse_fano_lp`

Common protocol print helpers:

- `print_alice_guessing_metrics()`
- `print_alice_uncertainty_metrics()`
- `print_eve_guessing_metrics_lp()`
- `print_eve_uncertainty_metrics_reverse_fano_lp()`
- `print_key_rate_summary_reverse_fano_lp()`

## File Map

- `contextualityqkd/scenario.py`: validated Bob-outcome scenario container
- `contextualityqkd/contextuality.py`: contextuality LPs and assignment-ray machinery
- `contextualityqkd/randomness.py`: Eve LP backend + entropy helpers
- `contextualityqkd/protocol.py`: protocol metrics/reporting built on a scenario
- `contextualityqkd/demos/`: four kept QKD demos + `run_all.py`
- `tests/`: scenario/protocol unit tests

## Dependencies

Main runtime dependencies include:

- `numpy`
- `sympy`
- `mosek`
- `scipy`
- `pycddlib`
- `methodtools`

See `pyproject.toml` and/or `requirements.txt` for full dependency declarations.

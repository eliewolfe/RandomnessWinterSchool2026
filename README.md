# Randomness Winter School 2026

`contextualityqkd` is a research-oriented Python package for:

- building prepare-and-measure contextuality scenarios from quantum or GPT descriptions,
- discovering or validating operational equivalences (OPEQs),
- quantifying certified randomness through MOSEK linear programs, and
- quantifying contextuality via robustness to dephasing noise.

The package is organized around one central object (`ContextualityScenario`) and two optimization workflows:

1. **Randomness certification** (Eve guessing LP).
2. **Contextuality quantification** (dephasing robustness LP after extremal-ray enumeration).

## Intended Use Cases

This package is intended for users who want to:

- study device-independent or semi-device-independent randomness from contextuality constraints,
- move between quantum matrix descriptions and GPT-vector descriptions,
- infer measurement contexts from a flat effect list,
- inspect and manipulate operational equivalences directly, and
- compare different scenarios through a contextuality metric (`r*`, robustness to dephasing).

## Installation

### Recommended (Conda)

```bash
conda create -n py13 python=3.13 -y
conda activate py13
conda install -y numpy scipy sympy
pip install mosek pycddlib
```

### Alternative (pip-only)

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Notes:

- `mosek` is required for LP solves (randomness and robustness). You need a working MOSEK license.
- `pycddlib` is used by default for extremal-ray enumeration in contextuality robustness.
- The package also includes MOSEK-based cone conversion/enumeration utilities in `extremal_finders.py`.

## Quick Start

Run split demos from `contextualityqkd/demos/`:

```bash
python -m contextualityqkd.demos.randomness_qubit_z_x_xplusz
python -m contextualityqkd.demos.qkd_qubit_z_x_xplusz
python -m contextualityqkd.demos.qkd_porac_3_2
```

Or run all demos serially:

```bash
python -m contextualityqkd.demos.run_all
```

## Conceptual Pipeline

High-level flow:

1. Prepare data as `P(a,b|x,y)` and OPEQs, or construct these from quantum/GPT inputs.
2. Build a `ContextualityScenario`.
3. Run one or both analyses through scenario methods:
   - `scenario.compute_eve_guessing_table(...)`
   - `scenario.compute_keyrate_table(...)`
   - `scenario.compute_dephasing_robustness(...)`
   - `scenario.compute_contextual_fraction(...)`

## Data and OPEQ Conventions

### Behavior table convention

The package uses:

- `data[x, y, a, b] = P(a,b|x,y)`
- padded shape `(X, Y, A_max, B_max)` internally

Indices:

- `x`: preparation setting
- `y`: measurement setting
- `a`: source/preparation outcome
- `b`: measurement outcome

Input options:

- dense input with explicit padded shape `(X, Y, A_max, B_max)`, or
- ragged nested lists `data[x][y][a][b]` under the model `A=A(x)`, `B=B(y)`.

When ragged input is used, the package pads with zeros and tracks
`a_cardinality_per_x` and `b_cardinality_per_y`.

Normalization requirement:

- for each `(x,y)`, `sum_{a,b} P(a,b|x,y) = 1`.

Special case:

- if `A = 1`, this reduces to a single-outcome-per-setting source behavior 
  a.k.a. controlled preperations, and is often read as `p(b|x,y)`.

### Operational-equivalence convention

Preparation OPEQs:

- shape `(N_prep, X, A_max)` (or `(X,A_max)` for a single OPEQ) after padding,
- each OPEQ coefficient array `c[x,a]` must satisfy:
  - `sum_{x,a} c[x,a] P(a,b|x,y) = 0` for all `(y,b)`.

Measurement OPEQs:

- shape `(N_meas, Y, B_max)` (or `(Y,B_max)` for a single OPEQ) after padding,
- each OPEQ coefficient array `d[y,b]` must satisfy:
  - `sum_{y,b} d[y,b] P(a,b|x,y) = 0` for all `(x,a)`.

For ragged scenarios, padded coordinates are always enforced as structural zeros by
automatically injected OPEQ rows.

These are exactly the linear constraints used by the LPs.

## Why Constructors from Quantum and GPT Inputs?

In practice, users often start from:

- quantum objects (density matrices and effects), or
- GPT vectors (states/effects in an operational vector space),

not from a fully assembled `(X,Y,A,B)` table with explicit OPEQs.

The constructors automate:

1. conversion from input representation to GPT vectors,
2. measurement-context detection from flat effect lists,
3. behavior table construction `P(a,b|x,y)`, and
4. OPEQ discovery from nullspaces.

This reduces boilerplate and ensures consistent conventions.

## Core API by Module

### `scenario.py`

`ContextualityScenario` is the canonical container used by all optimization routines.

Core responsibilities:

- stores validated `data`, `opeq_preps`, `opeq_meas`,
- supports ragged inputs and internally pads to dense arrays,
- discovers missing OPEQs (nullspace-based),
- validates provided OPEQs,
- exposes cardinalities (`X,Y,A_max,B_max`) plus per-setting cardinalities,
- exposes validity masks (`valid_a_mask`, `valid_b_mask`, `valid_ab_mask`),
- offers printing/formatting utilities,
- computes Alice-side guessing benchmarks:
  - `alice_optimal_guessing_bob_probability` (cached `(X,Y)` table)
  - `alice_optimal_average_guessing_bob_probability`

Constructor notes:

- `ContextualityScenario(data=...)` infers per-setting cardinalities from ragged input directly, from trailing-zero support in dense padded arrays, or from masks in dense `numpy.ma.MaskedArray` input.

Important behavior:

- If `verbose=True` and OPEQs are omitted, the class emits warnings that OPEQs are being discovered from data.

### `quantum.py`

Bridges quantum and GPT representations and builds scenarios from either.

Important functions:

- Quantum helpers:
  - `projector(ket)`
  - `projector_hs_vector(ket)`
  - `gell_mann_matrices(d)`
  - `convert_matrix_list_to_vector_list(...)`
  - `convert_matrix_to_vector(...)`
  - `direct_probability_table_from_quantum(...)`
  - `probability_table_from_quantum_via_gpt(...)`
- GPT helpers:
  - `probability_table_from_gpt_vectors(...)` (supports `return_masked`; default auto-masks mixed-cardinality grouped inputs)
  - `discover_operational_equivalences_from_gpt_objects(...)`
  - `infer_measurements_from_gpt_effect_set(...)`
  - `data_table_from_gpt_states_and_effect_set(...)`
- High-level scenario classes:
  - `GPTContextualityScenario(...)`
  - `QuantumContextualityScenario(...)`

Projective fast path:

- `QuantumContextualityScenario(...)` detects when all states/effects are projectors and (if no custom basis/unit effect is supplied) uses a direct Hilbert-Schmidt vectorization path instead of Gell-Mann expansion.

### `randomness.py`

Contains internal MOSEK LP backends for Eve's guessing probability and entropy helpers.

Main backend functions:

- `eve_optimal_guessing_probability(scenario, x, y, guess_who="Bob")`
- `eve_optimal_average_guessing_probability(scenario, guess_who="Bob")`
- `analyze_scenario(scenario, guess_who="Bob")` (fills Eve guessing and key-rate tables)
- `min_entropy(p_guess)`

Notes:

- For user workflows, prefer scenario methods:
  - `scenario.compute_eve_guessing_table(...)`
  - `scenario.compute_eve_average_guessing_probability(...)`
  - `scenario.compute_keyrate_table(...)`

### `contextuality.py`

Contains internal simplex-embeddability workflow, dephasing-robustness, and contextual-fraction LPs.

Main functions:

- `preparation_assignment_extremals(scenario)`
- `effect_assignment_extremals(scenario)`
- `assess_simplex_embeddability(scenario, ...)`
- `contextuality_robustness_to_dephasing(scenario, ...)`
- `assess_contextual_fraction(scenario, ...)`
- `noncontextual_fraction(scenario, ...)`
- `contextual_fraction(scenario, ...)`

Result dataclass:

- `SimplexEmbeddabilityResult` with:
  - `is_simplex_embeddable`
  - `dephasing_robustness`
  - extremals, optional coupling weights, and solver status.
- `ContextualFractionResult` with:
  - `noncontextual_fraction`
  - `contextual_fraction`
  - extremals, optional coupling weights, and solver status.

Notes:

- For user workflows, prefer scenario methods:
  - `scenario.compute_dephasing_robustness(...)`
  - `scenario.compute_contextual_fraction(...)`
  - `scenario.compute_noncontextual_fraction(...)`

### `linalg_utils.py`

Shared linear algebra utilities:

- `null_space_basis(...)` with methods:
  - `"sympy"` (default)
  - `"numpy"`
  - `"scipy"`
- `select_linearly_independent_rows(...)`
- `enumerate_cone_extremal_rays(...)` (CDD or MOSEK backend)

### `extremal_finders.py`

Cone representation conversion with two backends:

- CDD:
  - `cone_h_to_v_cdd(...)`
  - `cone_v_to_h_cdd(...)`
- MOSEK:
  - `cone_h_to_v_mosek(...)`
  - `cone_v_to_h_mosek(...)`

Conventions used there:

- H-rep: `A_ineq x >= 0`, `A_eq x = 0`
- V-rep: `Cone(rays) + Lin(lines)`

## Randomness LP: Physical Idea and Constraints

### Physical interpretation

The Eve LP asks:

- given observed behavior `P(a,b|x,y)` and OPEQs,
- what is the maximum probability that an adversary Eve can guess a chosen target (`Bob`, `Alice`, or `Both`) at chosen settings?

The LP introduces a tripartite extension `P_t(a,b,e|x,y)` (indexed by target `t` for averaging scenarios).

Eve's guess is encoded as `e` and the objective rewards:

- `e=b` when `guess_who="Bob"` (default),
- `e=a` when `guess_who="Alice"`,
- `e=(a,b)` when `guess_who="Both"`.

### Why OPEQ constraints involve grouped parties

Preparation OPEQs constrain what can be signaled from the source side. In the LP they are enforced while grouping Bob+Eve together:

- for each preparation OPEQ `c[x,a]`:
  - `sum_{x,a} c[x,a] P_t(a,b,e|x,y) = 0` for all `(y,b,e)`.

Measurement OPEQs are dual: they are enforced while grouping Alice+Eve together:

- for each measurement OPEQ `d[y,b]`:
  - `sum_{y,b} d[y,b] P_t(a,b,e|x,y) = 0` for all `(x,a,e)`.

This is the operational requirement that Eve's side information remains compatible with both sets of observed equivalence relations.

### LP structure

Variables:

- nonnegative `P_t(a,b,e|x,y)`.

Constraints:

1. Data consistency:
   - `sum_e P_t(a,b,e|x,y) = P(a,b|x,y)`.
2. Preparation OPEQ constraints (grouping Bob+Eve).
3. Measurement OPEQ constraints (grouping Alice+Eve).

Objective:

- maximize Eve's success probability at target pair(s):
  - single-target: one `(x,y)`,
  - average-target: uniform average across all `(x,y)`.

Outputs:

- `p_guess` in `[0,1]`,
- convert to min-entropy via `H_min = -log2(p_guess)`.

## Alice Guessing Benchmark

`ContextualityScenario` includes built-in non-LP benchmarks:

- `alice_optimal_guessing_bob_probability`
- `alice_optimal_average_guessing_bob_probability`
- `bob_optimal_guessing_alice`
- `bob_optimal_average_guessing_alice_probability`
- `largest_joint_probability`

At fixed `(x,y)`, Alice knows `a`, so she can pick best `b` per `a`:

- `sum_a p(a|x,y) max_b p(b|x,y,a)`
- implemented equivalently as `sum_a max_b P(a,b|x,y)`.

This is useful to contrast internal source-side knowledge with adversarially constrained Eve randomness.

## Robustness to Dephasing: High-Level Algorithm

The contextuality measure implemented is dephasing robustness `r*`:

- smallest `r in [0,1]` such that:
  - `(1-r) P + r D`
  - is representable by a noncontextual simplex-embeddable model.

Here `D` is a dephasing target behavior:

- default is built from marginals (`P(a|x)` and averaged `Q(b|y)`),
- you can also pass a custom `dephasing_target`.

Algorithm steps:

1. Build preparation-assignment cone:
   - variables `p(x,a) >= 0`, plus preparation OPEQ equalities.
2. Build effect-assignment cone:
   - variables `q(y,b) >= 0`, plus measurement OPEQ equalities.
3. Enumerate extremal rays of both cones (default CDD backend).
4. Solve one MOSEK LP over nonnegative coupling weights `w_ij` and scalar `r`:
   - reconstruct dephased behavior from convex combination of ray products,
   - minimize `r`.

Interpretation used in demo:

- larger `r*` means more contextuality (more dephasing needed to reach noncontextual explainability).
- `r*` near 0 indicates simplex embeddability already (or nearly) present.

## Contextual Fraction: High-Level Algorithm

The contextual fraction workflow solves for the largest noncontextual subbehavior mass `lambda*`:

- `S(a,b|x,y) = sum_ij w_ij R_ij(a,b|x,y)` where `R_ij` are prep/effect assignment-ray products and `w_ij >= 0`.
- constraints:
  - `S(a,b|x,y) <= P(a,b|x,y)` for all entries (subbehavior inequality),
  - `sum_ab S(a,b|x,y) = lambda` for all `(x,y)` (uniform subnormalization mass).
- objective: maximize `lambda`.

Outputs:

- `noncontextual_fraction = lambda*`
- `contextual_fraction = 1 - lambda*`
- post-solve sanity check verifies `lambda*` is within `[0, 1]` up to tolerance.

Implementation note:

- This is solved directly in the cone-weight representation; no explicit normalized-vertex enumeration is required.

## Demo Guide (`contextualityqkd/demos/*.py`)

Each demo now has its own script:

- Randomness demos:
  - `contextualityqkd/demos/randomness_qubit_z_x_xplusz.py`
  - `contextualityqkd/demos/randomness_qubit_xplusz_xminusz.py`
  - `contextualityqkd/demos/randomness_hexagon_povm.py`
  - `contextualityqkd/demos/randomness_cabello_18ray.py`
  - `contextualityqkd/demos/randomness_peres_24ray.py`
- QKD demos:
  - `contextualityqkd/demos/qkd_qubit_z_x_xplusz.py`
  - `contextualityqkd/demos/qkd_qubit_xplusz_xminusz.py`
  - `contextualityqkd/demos/qkd_hexagon_povm.py`
  - `contextualityqkd/demos/qkd_cabello_18ray.py`
  - `contextualityqkd/demos/qkd_peres_24ray.py`
  - `contextualityqkd/demos/qkd_porac_3_2.py` (PORAC (3,2), Eq. (22)-style reporting)
- Optional runner:
  - `python -m contextualityqkd.demos.run_all`

## Minimal Usage Patterns

### 1) Quantum input to randomness

```python
from contextualityqkd.quantum import QuantumContextualityScenario

scenario = QuantumContextualityScenario(
    quantum_states=quantum_states,         # grouped (X,A,d,d) or flat (N,d,d)+indices
    quantum_effects=effect_set,
    infer_measurement_indices=True,
    outcomes_per_measurement=2,
    verbose=True,
)

measurement_indices = scenario.measurement_indices
p_guess_table = scenario.compute_eve_guessing_table(guess_who="Bob")
p_guess_target = p_guess_table[0, 0]
```

### 2) GPT input to scenario

```python
from contextualityqkd.quantum import GPTContextualityScenario

scenario = GPTContextualityScenario(
    gpt_states=gpt_states,              # grouped/ragged or flat + preparation_indices
    gpt_effects=gpt_effect_set,         # grouped/ragged or flat + measurement_indices
    infer_measurement_indices=True,     # optional flat-effect inference path
    outcomes_per_measurement=2,
)

measurement_indices = scenario.measurement_indices
```

### 3) Randomness metrics

```python
import numpy as np

p_target = scenario.compute_eve_guessing_table(guess_who="Bob")[0, 0]
p_avg = scenario.compute_eve_average_guessing_probability(guess_who="Bob")
hmin = float(-np.log2(p_target))
```

### 4) Contextuality robustness

```python
r_star = scenario.compute_dephasing_robustness()
cf = scenario.compute_contextual_fraction()
```

## Project Structure

- `contextualityqkd/`
  - `scenario.py`: scenario container, OPEQ discovery/validation, Alice benchmark
  - `quantum.py`: quantum/GPT conversions and scenario constructors
  - `randomness.py`: internal Eve LP backends and entropy helpers
  - `contextuality.py`: internal simplex-embeddability/contextuality LP backends
  - `linalg_utils.py`: nullspace/independence/extremal-ray helper wrappers
  - `extremal_finders.py`: CDD/MOSEK cone conversion backends
- `contextualityqkd/demos/`: split per-example randomness/QKD demos, including PORAC (3,2)
- `pyproject.toml`: metadata and dependencies
- `requirements.txt`: runtime dependencies

## Practical Notes

- Measurement inference from a flat effect set is combinatorial in the number of effects; constrain with `outcomes_per_measurement` when possible.
- Mixed inferred measurement cardinalities are supported; returned grouped effects are zero-padded internally.
- If you already know measurement contexts, pass `measurement_indices` into `GPTContextualityScenario` / `QuantumContextualityScenario`.
- `QuantumContextualityScenario(...)` automatically uses a projector fast path when applicable.
- For advanced cone work, import directly from `contextualityqkd.extremal_finders`.
- For strict consistency checks, call `scenario.sanity_check()`.

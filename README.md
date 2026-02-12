# Randomness Winter School 2026

`randomness_contextuality_lp` is a research-oriented Python package for:

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

Run the full demonstration script:

```bash
python demo.py
```

See `demo.py` for five end-to-end examples, from simple qubit scenarios to Cabello and Peres-style constructions.

## Conceptual Pipeline

High-level flow:

1. Prepare data as `P(a,b|x,y)` and OPEQs, or construct these from quantum/GPT inputs.
2. Build a `ContextualityScenario`.
3. Run one or both analyses:
   - `eve_optimal_guessing_probability` / `eve_optimal_average_guessing_probability`
   - `contextuality_robustness_to_dephasing`

## Data and OPEQ Conventions

### Behavior table convention

The package uses:

- `data[x, y, a, b] = P(a,b|x,y)`
- shape `(X, Y, A, B)`

Indices:

- `x`: preparation setting
- `y`: measurement setting
- `a`: source/preparation outcome
- `b`: measurement outcome

Normalization requirement:

- for each `(x,y)`, `sum_{a,b} P(a,b|x,y) = 1`.

Special case:

- if `A = 1`, this reduces to a single-outcome-per-setting source behavior 
  a.k.a. controlled preperations, and is often read as `p(b|x,y)`.

### Operational-equivalence convention

Preparation OPEQs:

- shape `(N_prep, X, A)` (or `(X,A)` for a single OPEQ),
- each OPEQ coefficient array `c[x,a]` must satisfy:
  - `sum_{x,a} c[x,a] P(a,b|x,y) = 0` for all `(y,b)`.

Measurement OPEQs:

- shape `(N_meas, Y, B)` (or `(Y,B)` for a single OPEQ),
- each OPEQ coefficient array `d[y,b]` must satisfy:
  - `sum_{y,b} d[y,b] P(a,b|x,y) = 0` for all `(x,a)`.

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
- discovers missing OPEQs (nullspace-based),
- validates provided OPEQs,
- exposes cardinalities (`X,Y,A,B`),
- offers printing/formatting utilities,
- computes Alice-side guessing benchmarks:
  - `alice_optimal_guessing_probability(x,y)`
  - `alice_optimal_average_guessing_probability()`

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
  - `probability_table_from_gpt_vectors(...)`
  - `discover_operational_equivalences_from_gpt_objects(...)`
  - `infer_measurements_from_gpt_effect_set(...)`
  - `data_table_from_gpt_states_and_effect_set(...)`
- High-level constructors:
  - `contextuality_scenario_from_gpt(...)`
  - `contextuality_scenario_from_quantum(...)`

Projective fast path:

- `contextuality_scenario_from_quantum(...)` detects when all states/effects are projectors and (if no custom basis/unit effect is supplied) uses a direct Hilbert-Schmidt vectorization path instead of Gell-Mann expansion.

### `randomness.py`

Contains MOSEK LPs for Eve's guessing probability and convenience runners.

Main functions:

- `eve_optimal_guessing_probability(scenario, x, y)`
- `eve_optimal_average_guessing_probability(scenario)`
- `min_entropy_bits(p_guess)`
- `run_gpt_example(...)`
- `run_quantum_example(...)`

### `contextuality.py`

Contains simplex-embeddability and dephasing-robustness workflow.

Main functions:

- `preparation_assignment_extremals(scenario)`
- `effect_assignment_extremals(scenario)`
- `assess_simplex_embeddability(scenario, ...)`
- `contextuality_robustness_to_dephasing(scenario, ...)`

Result dataclass:

- `SimplexEmbeddabilityResult` with:
  - `is_simplex_embeddable`
  - `dephasing_robustness`
  - extremals, optional coupling weights, and solver status.

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
- what is the maximum probability that an adversary Eve can guess Bob's output at chosen settings?

The LP introduces a tripartite extension `P_t(a,b,e|x,y)` (indexed by target `t` for averaging scenarios).

Eve's guess is encoded as `e` and the objective rewards events with `e=b` at target settings.

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

`ContextualityScenario` includes a built-in non-LP benchmark:

- `alice_optimal_guessing_probability(x,y)`
- `alice_optimal_average_guessing_probability()`

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

## Demo Guide (`demo.py`)

`demo.py` is the best practical tour of the package.

It currently demonstrates:

1. **Example 1**: qubit Z/X/(X+Z), with inferred measurements and targeted randomness.
2. **Example 2**: qubit (X+Z)/(X-Z), contrasting randomness with Example 1.
3. **Example 3**: hexagon GPT construction with `|A|=2` (multi-outcome source), manual grouped preparations/effects.
4. **Example 4**: Cabello-style 18-ray GPT construction, repeated effects across contexts, manual grouping.
5. **Example 5**: Peres 24-ray construction restricted to the 6 Mermin-square contexts (disjoint 4-outcome bases).

Helper functions used in demo:

- `_print_manual_target_randomness(...)`:
  - prints Eve and Alice target guessing probabilities.
- `_print_manual_target_robustness(...)`:
  - prints `r*` robustness and interpretation.
- `_print_measurement_index_sets(...)`:
  - prints manually supplied measurement context indices.

The demo uses both:

- high-level convenience (`run_quantum_example`), and
- explicit low-level construction (`ContextualityScenario(...)` from manually computed GPT objects).

## Minimal Usage Patterns

### 1) Quantum input to randomness

```python
from randomness_contextuality_lp.randomness import run_quantum_example

scenario, measurement_indices, p_guess = run_quantum_example(
    quantum_states=quantum_states,      # (X,d,d) or (X,A,d,d)
    quantum_effect_set=effect_set,      # (N_effects,d,d), flat set
    target_pair=(0, 0),
    outcomes_per_measurement=2,
    verbose=True,
)
```

### 2) GPT input to scenario

```python
from randomness_contextuality_lp.quantum import contextuality_scenario_from_gpt

scenario, measurement_indices = contextuality_scenario_from_gpt(
    gpt_states=gpt_states,              # (X,K) or (X,A,K)
    gpt_effect_set=gpt_effect_set,      # (N_effects,K)
    outcomes_per_measurement=2,
    return_measurement_indices=True,
)
```

### 3) Randomness metrics

```python
from randomness_contextuality_lp.randomness import (
    eve_optimal_guessing_probability,
    eve_optimal_average_guessing_probability,
    min_entropy_bits,
)

p_target = eve_optimal_guessing_probability(scenario, x=0, y=0)
p_avg = eve_optimal_average_guessing_probability(scenario)
hmin = min_entropy_bits(p_target)
```

### 4) Contextuality robustness

```python
from randomness_contextuality_lp.contextuality import contextuality_robustness_to_dephasing

r_star = contextuality_robustness_to_dephasing(scenario)
```

## Project Structure

- `randomness_contextuality_lp/`
  - `scenario.py`: scenario container, OPEQ discovery/validation, Alice benchmark
  - `quantum.py`: quantum/GPT conversions and scenario constructors
  - `randomness.py`: Eve LPs and convenience runners
  - `contextuality.py`: simplex embeddability and robustness LP
  - `linalg_utils.py`: nullspace/independence/extremal-ray helper wrappers
  - `extremal_finders.py`: CDD/MOSEK cone conversion backends
- `demo.py`: end-to-end examples
- `pyproject.toml`: metadata and dependencies
- `requirements.txt`: runtime dependencies

## Practical Notes

- Measurement inference from a flat effect set is combinatorial in the number of effects; constrain with `outcomes_per_measurement` when possible.
- If you already know measurement contexts, you can bypass automatic detection and directly build grouped objects (as in `demo.py` Examples 3 and 4).
- If you want the projector-detection fast path, call `contextuality_scenario_from_quantum(...)` directly (the convenience `run_quantum_example(...)` currently uses the generic conversion path).
- For advanced cone work, import directly from `randomness_contextuality_lp.extremal_finders`.
- For strict consistency checks, call `scenario.sanity_check()`.

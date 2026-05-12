# contextualityqkd: Bob-Outcome QKD from Contextuality

`contextualityqkd` analyzes prepare-and-measure QKD protocols where:

- Alice chooses preparation `x`
- Bob chooses measurement `y` and gets outcome `b`
- Eve is constrained by the observed behavior `p(b|x,y)` plus operational equivalences (OPEQs)

The package separates:

1. **Scenario construction** (`ContextualityScenario`, `GPTContextualityScenario`, `QuantumContextualityScenario`)
2. **Protocol analysis** (`ContextualityProtocol`)

This separation is central: one scenario can be reused for many protocol choices (`where_key`).

## Current API At A Glance

### Core classes

- `ContextualityScenario`: generic validated Bob-outcome behavior container
- `GPTContextualityScenario`: scenario from GPT vectors/states/effects
- `QuantumContextualityScenario`: scenario from quantum matrices, converted to GPT internally
- `ContextualityProtocol`: Alice/Eve metrics and key-rate analysis on top of a scenario

### Main constructor/factory pipelines

- Generic scenario from data:
  - `ContextualityScenario(data, opeq_preps=None, opeq_meas=None, atol=..., verbose=...)`
- GPT scenario (direct):
  - `GPTContextualityScenario(gpt_states=..., gpt_effects=..., measurement_indices=..., ...)`
- GPT factory helpers:
  - `GPTContextualityScenario.from_xz_ring(...)`
  - `GPTContextualityScenario.from_integer_rays(...)`
- Quantum scenario (direct):
  - `QuantumContextualityScenario(quantum_states=..., quantum_effects=..., ...)`
- Quantum factory helper:
  - `QuantumContextualityScenario.from_quantum_states_effects(...)`

## Protocol Construction Pipelines

## 1) Manual `where_key` protocol

Use this when you already know the key-eligible preparations for each measurement:

```python
from contextualityqkd.protocol import ContextualityProtocol

protocol = ContextualityProtocol(
    scenario,
    where_key=measurement_indices,  # one row per y
)
```

## 2) Automatic `where_key` optimization (built in)

Use this to automatically sweep admissible key sets and choose the best stage by reverse-Fano bits per experimental run:

```python
from contextualityqkd.protocol import ContextualityProtocol

protocol = ContextualityProtocol(
    scenario,
    where_key="Automatic",  # also accepts "auto", case-insensitive
    optimize_verbose=True,
)
```

### Automatic optimizer options

`ContextualityProtocol(..., where_key="Automatic", ...)` supports:

- `optimize_cluster_tolerance` (default `1e-6`)
- `optimize_cluster_by`
  - `"threshold_uncertainty"` (default)
  - `"threshold_alice_guess_bob_probability"`
- `atol` (protocol tolerance; used for stage-score tie handling)
- `optimize_tie_break`
  - `"earliest_optimal_stage"` (default)
  - `"latest_optimal_stage"`

### Automatic optimization results

When automatic mode is used:

- `protocol.where_key_optimization_result` returns full report dictionary
- top-level keys include:
  - `mode`, `objective`, `cluster_by`, `cluster_tolerance`, `tie_tolerance`, `tie_break`, `total_stages`, `stages`, `best_stage`
- each stage includes:
  - `stage_index`, `where_key`, `key_counts_by_y`, `is_uniform_key_count`, `uniform_key_count`
  - `threshold_uncertainty`, `threshold_alice_guess_bob_probability`
  - `key_generation_probability_per_run`, `bits_per_key_generating_run`, `bits_per_experimental_run`

## Printing Helpers

### Scenario printing

`scenario.print_probabilities(...)` now prints its own header by default (numeric/symbolic), so demos do not need a separate title line for the probability block.

Key methods:

- `scenario.print_probabilities(...)`
- `scenario.print_operational_equivalences(...)`
- `scenario.print_contextuality_measures(...)`

### Protocol printing

Key methods:

- `protocol.print_alice_guessing_metrics()`
- `protocol.print_alice_uncertainty_metrics()`
- `protocol.print_eve_guessing_metrics_lp()`
- `protocol.print_eve_uncertainty_metrics_reverse_fano_lp()`
- `protocol.print_key_rate_summary_reverse_fano_lp()`
- `protocol.print_where_key_optimization_best_stage(...)`

## Demo Guide (Updated)

All demos live in `contextualityqkd/demos/`.

## `qkd_xz_ring.py`

This demo is now a configurable ring sweep tool (majorly changed from earlier versions).

What it does:

- exposes two top-level knobs:
  - `NUM_STATES`
  - `NUM_MEAS`
- enforces `NUM_STATES % (2 * NUM_MEAS) == 0`
- builds evenly spaced antipodal deterministic measurement pairs from those settings
- constructs scenario via `GPTContextualityScenario.from_xz_ring(...)`
- runs automatic `where_key` optimization with verbose stage logging
- prints automatic best-stage identification and thresholds
- prints Alice/Eve metrics and reverse-Fano key-rate summaries for the selected automatic stage

This file is now the main ring exploration demo.

## `qkd_hexagon_projective.py`

What it does:

- builds a 6-state hexagon GPT scenario via `from_xz_ring`
- runs a manual protocol using deterministic pairs (`where_key=measurement_indices`)
- runs an additional automatic optimization stage and prints best-stage summary

## `qkd_peres_24ray.py`

What it does:

- builds Peres 24-ray GPT scenario via `from_integer_rays`
- runs manual basis-aligned key protocol (`where_key=measurement_indices`)
- runs automatic optimization stage and prints best-stage summary

## `qkd_cabello_18ray.py`

What it does:

- builds Cabello 18-ray GPT scenario via `from_integer_rays`
- runs manual context-aligned key protocol
- runs automatic optimization stage and prints best-stage summary

## `qkd_porac_3_2.py`

What it does:

- builds true quantum PORAC scenario via `QuantumContextualityScenario.from_quantum_states_effects`
- validates discovered preparation-OPEQ subspace against article constraints
- runs baseline protocol with `where_key=None`
- runs automatic optimization stage and prints best-stage summary

## Running

Per project instructions, use conda env `py13`.

Example commands:

```powershell
C:\Users\elupu\miniconda3\Scripts\conda.exe run -n py13 python -m contextualityqkd.demos.qkd_xz_ring
C:\Users\elupu\miniconda3\Scripts\conda.exe run -n py13 python -m contextualityqkd.demos.qkd_hexagon_projective
C:\Users\elupu\miniconda3\Scripts\conda.exe run -n py13 python -m contextualityqkd.demos.qkd_peres_24ray
C:\Users\elupu\miniconda3\Scripts\conda.exe run -n py13 python -m contextualityqkd.demos.qkd_cabello_18ray
C:\Users\elupu\miniconda3\Scripts\conda.exe run -n py13 python -m contextualityqkd.demos.qkd_porac_3_2
```

Run tests:

```powershell
C:\Users\elupu\miniconda3\Scripts\conda.exe run -n py13 python -m unittest discover -s tests -v
```

## Minimal Example

```python
import numpy as np
from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.quantum import GPTContextualityScenario

measurement_indices = [(0, 3), (1, 4), (2, 5)]
scenario = GPTContextualityScenario.from_xz_ring(
    num_states=6,
    measurement_indices=measurement_indices,
    verbose=False,
)

# Automatic protocol sweep + best-stage selection
protocol = ContextualityProtocol(
    scenario,
    where_key="Automatic",
    optimize_verbose=False,
    optimize_tie_break="earliest_optimal_stage",
)

result = protocol.where_key_optimization_result
best_stage = result["best_stage"]

# Standard metric reports
protocol.print_alice_guessing_metrics()
protocol.print_eve_guessing_metrics_lp()
protocol.print_key_rate_summary_reverse_fano_lp()
```

## File Map

- `contextualityqkd/scenario.py`: generic scenario container and formatting
- `contextualityqkd/quantum.py`: GPT/quantum scenario classes and factory constructors
- `contextualityqkd/protocol.py`: protocol metrics, LP-derived key rates, automatic optimizer
- `contextualityqkd/contextuality.py`: contextuality LP metrics (contextual fraction, robustness)
- `contextualityqkd/randomness_lp.py`: Eve LP backend
- `contextualityqkd/demos/`: five maintained demos listed above

## Dependencies

Core runtime dependencies include:

- `numpy`
- `sympy`
- `scipy`
- `methodtools`
- `pycddlib`
- `mosek`

See `pyproject.toml` / `requirements.txt` for full dependency declarations.

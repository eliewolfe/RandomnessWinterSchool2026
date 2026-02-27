"""QKD protocol demo: configurable XZ ring with automatic where_key optimization."""

from __future__ import annotations

from pathlib import Path

import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.quantum import (
    GPTContextualityScenario,
)
from contextualityqkd.scenario import ContextualityScenario


NUM_STATES = 54
NUM_MEAS = 9


def _validate_configuration(num_states: int, num_meas: int) -> None:
    if int(num_states) < 4:
        raise ValueError("NUM_STATES must be at least 4.")
    if int(num_states) % 2 != 0:
        raise ValueError("NUM_STATES must be even so antipodal measurement pairs are well-defined.")
    if int(num_meas) <= 0:
        raise ValueError("NUM_MEAS must be positive.")
    if int(num_states) % (2 * int(num_meas)) != 0:
        raise ValueError("NUM_STATES must be an integer multiple of 2 * NUM_MEAS.")


def _build_measurement_indices(num_states: int, num_meas: int) -> tuple[tuple[int, int], ...]:
    """Build evenly spaced antipodal deterministic pairs on the ring."""
    step = int(num_states) // (2 * int(num_meas))
    return tuple(
        (int(m) * step, int(m) * step + int(num_states) // 2)
        for m in range(int(num_meas))
    )


def main() -> None:
    num_states = int(NUM_STATES)
    num_meas = int(NUM_MEAS)
    _validate_configuration(num_states, num_meas)

    np.set_printoptions(precision=6, suppress=True)
    ContextualityScenario.print_title(
        f"QKD Protocol: {num_states}-state XZ ring, {num_meas} measurements, automatic where_key optimization"
    )

    # Evenly spaced 2-outcome measurements from antipodal state/effect pairs.
    measurement_indices = _build_measurement_indices(num_states, num_meas)

    scenario = GPTContextualityScenario.from_xz_ring(
        num_states=num_states,
        measurement_indices=measurement_indices,
        verbose=False,
    )

    scenario.print_probabilities(as_p_b_given_x_y=True, precision=3, representation="numeric")
    scenario.print_operational_equivalences(precision=3, representation="numeric")

    protocol = ContextualityProtocol(
        scenario,
        where_key="Automatic",
        optimize_verbose=True,
    )
    protocol.print_where_key_optimization_best_stage(leading_newline=True)
    protocol.print_alice_guessing_metrics()
    protocol.print_alice_uncertainty_metrics()
    protocol.print_eve_guessing_metrics_lp()
    protocol.print_eve_uncertainty_metrics_reverse_fano_lp()
    protocol.print_key_rate_summary_reverse_fano_lp()

    scenario.print_contextuality_measures(precision=3)


if __name__ == "__main__":
    main()

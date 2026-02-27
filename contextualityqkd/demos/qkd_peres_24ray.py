"""QKD protocol demo: Peres 24-ray construction with 6 disjoint bases."""

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


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    ContextualityScenario.print_title("QKD Protocol: Peres 24 rays in 6 disjoint 4-ray bases")

    rays = np.array(
        [
            [2, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 2],
            [1, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [1, -1, -1, 1],
            [1, -1, -1, -1],
            [1, -1, 1, 1],
            [1, 1, -1, 1],
            [1, 1, 1, -1],
            [1, 1, 0, 0],
            [1, -1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [0, 1, 0, -1],
            [1, 0, 1, 0],
            [1, 0, -1, 0],
            [1, 0, 0, -1],
            [1, 0, 0, 1],
            [0, 1, -1, 0],
            [0, 1, 1, 0],
        ],
        dtype=int,
    )

    measurement_indices = [tuple(range(4 * y, 4 * (y + 1))) for y in range(6)]

    scenario = GPTContextualityScenario.from_integer_rays(
        rays=rays,
        measurement_indices=measurement_indices,
        verbose=False,
    )
    protocol = ContextualityProtocol(scenario, where_key=measurement_indices)

    scenario.print_probabilities(as_p_b_given_x_y=True, precision=3, representation="symbolic")
    scenario.print_operational_equivalences(precision=3, representation="symbolic")
    protocol.print_alice_guessing_metrics()
    protocol.print_alice_uncertainty_metrics()
    protocol.print_eve_guessing_metrics_lp()
    protocol.print_eve_uncertainty_metrics_reverse_fano_lp()
    protocol.print_key_rate_summary_reverse_fano_lp()

    auto_protocol = ContextualityProtocol(
        scenario,
        where_key="Automatic",
        optimize_verbose=True,
    )
    auto_protocol.print_where_key_optimization_best_stage(leading_newline=True)

    scenario.print_contextuality_measures(precision=3)


if __name__ == "__main__":
    main()

"""QKD protocol demo: Cabello 18-ray Kochen-Specker construction."""

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
    ContextualityScenario.print_title("QKD Protocol: Cabello 18-ray KS set")

    labels = list("123456789ABCDEFGHI")
    rays = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, -1, -1, 1],
            [1, -1, -1, -1],
            [1, -1, 1, 1],
            [1, 1, 1, -1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [0, 1, 0, -1],
            [1, 0, -1, 0],
            [1, 0, 0, -1],
            [1, 0, 0, 1],
            [0, 1, -1, 0],
        ],
        dtype=int,
    )
    contexts = [
        "12BC",
        "13DE",
        "23GH",
        "45EF",
        "46GI",
        "56AB",
        "78AC",
        "79HI",
        "89DF",
    ]

    label_to_index = {lab: i for i, lab in enumerate(labels)}
    measurement_indices = [tuple(label_to_index[ch] for ch in context) for context in contexts]

    scenario = GPTContextualityScenario.from_integer_rays(
        rays=rays,
        measurement_indices=measurement_indices,
        verbose=False,
    )
    protocol = ContextualityProtocol(
        scenario,
        where_key=measurement_indices,
    )

    print("\nSymbolic probability table p(b|x,y):")
    scenario.print_probabilities(as_p_b_given_x_y=True, precision=3, representation="symbolic")
    scenario.print_operational_equivalences(precision=3, representation="symbolic")
    protocol.print_alice_guessing_metrics()
    protocol.print_alice_uncertainty_metrics()
    protocol.print_eve_guessing_metrics_lp()
    protocol.print_eve_uncertainty_metrics_reverse_fano_lp()
    protocol.print_key_rate_summary_reverse_fano_lp()

    scenario.print_contextuality_measures(precision=3)


if __name__ == "__main__":
    main()

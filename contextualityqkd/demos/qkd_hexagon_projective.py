"""QKD protocol demo: hexagon states with projective measurements only."""

from __future__ import annotations

from pathlib import Path

import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import sympy as sp

from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.quantum import (
    GPTContextualityScenario,
    projector_hs_vector,
    xz_plane_ket,
)
from contextualityqkd.scenario import ContextualityScenario


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    ContextualityScenario.print_title("QKD Protocol: Hexagon projective measurements")

    thetas = [sp.Integer(k) * sp.pi / 3 for k in range(6)]
    state_kets = [xz_plane_ket(theta) for theta in thetas]
    gpt_set = np.array([projector_hs_vector(ket) for ket in state_kets], dtype=object)

    measurement_indices = [(0, 3), (1, 4), (2, 5)]

    scenario = GPTContextualityScenario(
        gpt_states=gpt_set,
        gpt_effects=gpt_set,
        measurement_indices=measurement_indices,
        verbose=False,
    )
    protocol = ContextualityProtocol(scenario, where_key=measurement_indices)

    scenario.print_preparation_index_sets(tuple((x,) for x in range(scenario.X_cardinality)))
    scenario.print_measurement_index_sets(scenario.measurement_indices)
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

"""QKD protocol demo: hexagon states with projective measurements only."""

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
    ContextualityScenario.print_title("QKD Protocol: Hexagon projective measurements")

    measurement_indices = [(0, 3), (1, 4), (2, 5)]
    scenario = GPTContextualityScenario.from_xz_ring(
        num_states=6,
        measurement_indices=measurement_indices,
        verbose=False,
    )
    protocol = ContextualityProtocol(
        scenario=scenario,
        where_key=measurement_indices,
        master_key_holder="Alice",
        atol=1e-9,
        sdp_projective_bob=False,
        sdp_projective_eve=False,
        sdp_npa_level_bob=1,
        sdp_npa_level_eve=1,
        sdp_threads=None,
        sdp_verbose=0,
    )

    scenario.print_probabilities(as_p_b_given_x_y=True, precision=3, representation="symbolic")
    scenario.print_operational_equivalences(precision=3, representation="symbolic")
    protocol.print_alice_guessing_metrics()
    protocol.print_alice_uncertainty_metrics()
    protocol.print_eve_security_metrics(
        method="both",
        rate_type="reverse_fano",
        include_per_y_lp=False,
        precision_vector=3,
        precision_scalar=6,
        leading_newline=True,
    )

    # auto_protocol = ContextualityProtocol(
    #     scenario=scenario,
    #     where_key="Automatic",
    #     master_key_holder="Alice",
    #     atol=1e-9,
    #     optimize_cluster_tolerance=1e-6,
    #     optimize_cluster_by="threshold_uncertainty",
    #     optimize_tie_break="earliest_optimal_stage",
    #     sdp_projective_bob=False,
    #     sdp_projective_eve=False,
    #     sdp_npa_level_bob=1,
    #     sdp_npa_level_eve=1,
    #     sdp_threads=None,
    #     sdp_verbose=0,
    # )
    # auto_protocol.print_where_key_optimization_best_stage(leading_newline=True)

    try:
        scenario.print_contextuality_measures(precision=3)
    except ImportError as exc:
        print(f"\nContextuality measures skipped: {exc}")


if __name__ == "__main__":
    main()

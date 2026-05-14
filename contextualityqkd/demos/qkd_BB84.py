"""Pedagogical QKD demo: qubit Z/X with clustered preparations.

Recommended execution:
    python -m contextualityqkd.demos.qkd_BB84
"""

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
)
from contextualityqkd.scenario import ContextualityScenario

RUN_SDP = True


def main() -> None:
    # Keep numerical arrays readable while preserving enough detail.
    np.set_printoptions(precision=6, suppress=True)
    ContextualityScenario.print_title("QKD: Z and Xmeasurements")

    # ---------------------------------------------------------------------
    # 1) Define the qubit states/effects in ket form on the X-Z great circle.
    #    - 0 and pi are computational basis |0>,|1> (Z measurement basis)
    #    - +/- pi/2 are |+>,|-> (X basis)
    # ---------------------------------------------------------------------
    ket0 = GPTContextualityScenario.xz_plane_ket(0)
    ket1 = GPTContextualityScenario.xz_plane_ket(sp.pi)
    ket_plus = GPTContextualityScenario.xz_plane_ket(sp.pi / 2)
    ket_minus = GPTContextualityScenario.xz_plane_ket(-sp.pi / 2)
    
    state_kets = [ket0, ket1, ket_plus, ket_minus]
    effect_kets = [ket0, ket1, ket_plus, ket_minus]

    # ---------------------------------------------------------------------
    # 2) Specify preparation and measurement groupings explicitly.
    #    In this QKD-oriented version, preparations are clustered into pairs.
    # ---------------------------------------------------------------------
    preparation_indices = [(0, 1), (2, 3)]
    measurement_indices = [(0, 1), (2, 3)]

    # Expose the grouping decisions in the output.
    print("\nProvided preparation index sets:")
    for x, idx in enumerate(preparation_indices):
        print(f"x={x}: preparations {tuple(idx)}")
    print("\nProvided measurement index sets:")
    for y, idx in enumerate(measurement_indices):
        print(f"y={y}: effects {tuple(idx)}")

    # ---------------------------------------------------------------------
    # 3) Convert projectors -> GPT vectors.
    # ---------------------------------------------------------------------
    gpt_state_set = np.array([GPTContextualityScenario.projector_hs_vector(ket) for ket in state_kets], dtype=object)
    gpt_effect_set = np.array([GPTContextualityScenario.projector_hs_vector(ket) for ket in effect_kets], dtype=object)

    # ---------------------------------------------------------------------
    # 4) Build the scenario directly from GPT primitives.
    # ---------------------------------------------------------------------
    scenario = GPTContextualityScenario(
        gpt_states=gpt_state_set,
        gpt_effects=gpt_effect_set,
        measurement_indices=measurement_indices,
        verbose=False,
    )

    # ---------------------------------------------------------------------
    # 5) Print core structural objects and analysis outputs.
    # ---------------------------------------------------------------------
    scenario.print_measurement_operational_equivalences(precision=3, representation="symbolic")
    print("\nSymbolic probability table P(a,b|x,y):")
    scenario.print_probabilities(precision=3, representation="symbolic")

    protocol = ContextualityProtocol(
        scenario,
        where_key=None,
        master_key_holder="Bob",
        sdp_projective_bob=True,
        sdp_projective_eve=True,
        sdp_threads=1,
    )
    protocol.print_eve_guessing_metrics(method="lp")
    protocol.print_eve_uncertainty_metrics(method="lp")
    protocol.print_key_rate_summary(method="lp")
    if RUN_SDP:
        protocol.print_eve_guessing_metrics(method="sdp")
        protocol.print_eve_uncertainty_metrics(method="sdp")
        protocol.print_key_rate_summary(method="sdp")
    try:
        scenario.print_contextuality_measures(precision=3)
    except ImportError as exc:
        print(f"\nContextuality measures skipped: {exc}")


if __name__ == "__main__":
    main()

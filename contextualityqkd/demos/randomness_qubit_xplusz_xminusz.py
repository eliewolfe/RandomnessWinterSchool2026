"""Pedagogical randomness demo: qubit (X+Z)/(X-Z) with singleton preparations.

Recommended execution:
    python -m contextualityqkd.demos.randomness_qubit_xplusz_xminusz
"""

from __future__ import annotations

from pathlib import Path

import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import sympy as sp

from contextualityqkd.quantum import (
    GPTContextualityScenario,
    projector_hs_vector,
    xz_plane_ket,
)
from contextualityqkd.scenario import ContextualityScenario


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    ContextualityScenario.print_title("Randomness: (X+Z) and (X-Z) measurements")

    # ---------------------------------------------------------------------
    # 1) Define qubit states and the two rotated measurement bases.
    # ---------------------------------------------------------------------
    ket0 = xz_plane_ket(0)
    ket1 = xz_plane_ket(sp.pi)
    ket_plus = xz_plane_ket(sp.pi / 2)
    ket_minus = xz_plane_ket(-sp.pi / 2)
    ket_pz_plus = xz_plane_ket(sp.pi / 4)
    ket_pz_minus = xz_plane_ket(5 * sp.pi / 4)
    ket_mz_plus = xz_plane_ket(3 * sp.pi / 4)
    ket_mz_minus = xz_plane_ket(-sp.pi / 4)

    state_kets = [ket0, ket1, ket_plus, ket_minus]
    effect_kets = [ket_pz_plus, ket_pz_minus, ket_mz_plus, ket_mz_minus]

    # Singleton source outcomes for randomness-focused p(b|x,y) reporting.
    preparation_indices = [(0,), (1,), (2,), (3,)]
    measurement_indices = [(0, 1), (2, 3)]

    print("\nProvided preparation index sets:")
    for x, idx in enumerate(preparation_indices):
        print(f"x={x}: preparations {tuple(idx)}")
    print("\nProvided measurement index sets:")
    for y, idx in enumerate(measurement_indices):
        print(f"y={y}: effects {tuple(idx)}")

    # ---------------------------------------------------------------------
    # 2) Convert to GPT vectors.
    # ---------------------------------------------------------------------
    gpt_state_set = np.array([projector_hs_vector(ket) for ket in state_kets], dtype=object)
    gpt_effect_set = np.array([projector_hs_vector(ket) for ket in effect_kets], dtype=object)

    # ---------------------------------------------------------------------
    # 3) Build scenario directly from GPT primitives.
    # ---------------------------------------------------------------------
    scenario = GPTContextualityScenario(
        gpt_states=gpt_state_set,
        gpt_effects=gpt_effect_set,
        preparation_indices=preparation_indices,
        measurement_indices=measurement_indices,
        verbose=False,
    )

    # ---------------------------------------------------------------------
    # 4) Print core constraints plus randomness/contextuality diagnostics.
    # ---------------------------------------------------------------------
    scenario.print_measurement_operational_equivalences(precision=3, representation="symbolic")
    print("\nSymbolic probability table p(b|x,y):")
    scenario.print_probabilities(
        as_p_b_given_x_y=True,
        precision=3,
        representation="symbolic",
    )
    scenario.print_guessing_probability_grids(
        guess_who="Bob",
        precision=3,
        include_keyrate_pairs=False,
    )
    scenario.print_contextuality_measures(precision=3)


if __name__ == "__main__":
    main()

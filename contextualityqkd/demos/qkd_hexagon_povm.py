"""Pedagogical QKD demo: hexagon states with extra 3-outcome POVMs.

Recommended execution:
    python -m contextualityqkd.demos.qkd_hexagon_povm
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
    ContextualityScenario.print_title("QKD: Hexagon states/effects with added 3-outcome POVMs")

    # ---------------------------------------------------------------------
    # 1) Build six pure states/effects equally spaced in the X-Z plane.
    # ---------------------------------------------------------------------
    thetas = [sp.Integer(k) * sp.pi / 3 for k in range(6)]
    state_kets = [xz_plane_ket(theta) for theta in thetas]
    effect_kets = list(state_kets)

    # ---------------------------------------------------------------------
    # 2) Define grouped preparations and measurements.
    #    - First three measurements are binary pairs (0,3), (1,4), (2,5)
    #    - Last two measurements are 3-outcome POVMs built from rescaled effects
    # ---------------------------------------------------------------------
    preparation_indices = [(0, 3), (1, 4), (2, 5), (0, 2, 4), (1, 3, 5)]
    measurement_indices = [(0, 3), (1, 4), (2, 5), (6, 8, 10), (7, 9, 11)]
    povm_rescale = sp.Rational(2, 3)

    print("\nProvided preparation index sets:")
    for x, idx in enumerate(preparation_indices):
        print(f"x={x}: preparations {tuple(idx)}")
    print("\nProvided measurement index sets:")
    for y, idx in enumerate(measurement_indices):
        print(f"y={y}: effects {tuple(idx)}")

    # ---------------------------------------------------------------------
    # 3) Convert to GPT vectors.
    #    We duplicate and rescale effects to create explicit 3-outcome POVM slots.
    # ---------------------------------------------------------------------
    gpt_state_set = np.array([projector_hs_vector(ket) for ket in state_kets], dtype=object)
    gpt_effect_set_base = np.array([projector_hs_vector(ket) for ket in effect_kets], dtype=object)
    gpt_effect_set_povm = np.array([povm_rescale * effect for effect in gpt_effect_set_base], dtype=object)
    gpt_effect_set = np.concatenate([gpt_effect_set_base, gpt_effect_set_povm], axis=0)

    # ---------------------------------------------------------------------
    # 4) Assemble scenario directly from GPT primitives.
    # ---------------------------------------------------------------------
    scenario = GPTContextualityScenario(
        gpt_states=gpt_state_set,
        gpt_effects=gpt_effect_set,
        preparation_indices=preparation_indices,
        measurement_indices=measurement_indices,
        verbose=False,
    )

    # ---------------------------------------------------------------------
    # 5) Print symbolic objects and run built-in analyses.
    # ---------------------------------------------------------------------
    scenario.print_measurement_operational_equivalences(precision=3, representation="symbolic")
    print("\nSymbolic probability table P(a,b|x,y):")
    scenario.print_probabilities(precision=3, representation="symbolic")
    scenario.print_guessing_probability_grids(
        guess_who="Bob",
        precision=3,
        include_keyrate_pairs=True,
    )
    scenario.print_contextuality_measures(precision=3)


if __name__ == "__main__":
    main()

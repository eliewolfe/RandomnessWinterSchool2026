"""Pedagogical QKD demo: Cabello 18-ray Kochen-Specker construction.

Recommended execution:
    python -m contextualityqkd.demos.qkd_cabello_18ray
"""

from __future__ import annotations

from pathlib import Path

import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from contextualityqkd.quantum import (
    GPTContextualityScenario,
    normalize_integer_rays_symbolic,
    projector_hs_vector,
)
from contextualityqkd.scenario import ContextualityScenario


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    ContextualityScenario.print_title("QKD: Cabello 18-ray KS set")

    # ---------------------------------------------------------------------
    # 1) Define the Cabello 18-ray set and the 9 measurement contexts.
    #    Context strings use labels 1..9,A..I as compact ray identifiers.
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # 2) Convert integer rays into normalized symbolic kets and projectors.
    # ---------------------------------------------------------------------
    kets = normalize_integer_rays_symbolic(rays)
    gpt_set = np.array([projector_hs_vector(ket) for ket in kets], dtype=object)

    # Build explicit index maps for each context.
    label_to_index = {lab: i for i, lab in enumerate(labels)}
    preparation_indices = [tuple(label_to_index[ch] for ch in context) for context in contexts]
    measurement_indices = list(preparation_indices)

    print("\nProvided preparation index sets:")
    for x, idx in enumerate(preparation_indices):
        print(f"x={x}: preparations {tuple(idx)}")
    print("\nProvided measurement index sets:")
    for y, idx in enumerate(measurement_indices):
        print(f"y={y}: effects {tuple(idx)}")

    # ---------------------------------------------------------------------
    # 3) Build scenario directly from GPT primitives.
    # ---------------------------------------------------------------------
    scenario = GPTContextualityScenario(
        gpt_states=gpt_set,
        gpt_effects=gpt_set,
        preparation_indices=preparation_indices,
        measurement_indices=measurement_indices,
        verbose=False,
    )

    # ---------------------------------------------------------------------
    # 4) Report structural constraints and metrics.
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

"""Pedagogical QKD demo: Peres 24-ray construction with 6 disjoint bases.

Recommended execution:
    python -m contextualityqkd.demos.qkd_peres_24ray
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
    ContextualityScenario.print_title("QKD: Peres 24 rays in 6 disjoint 4-ray bases")

    # ---------------------------------------------------------------------
    # 1) Define Peres's 24 rays as integer representatives in R^4.
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # 2) Normalize rays, vectorize projectors, and choose 6 disjoint contexts.
    # ---------------------------------------------------------------------
    kets = normalize_integer_rays_symbolic(rays)
    gpt_set = np.array([projector_hs_vector(ket) for ket in kets], dtype=object)

    preparation_indices = [tuple(range(4 * x, 4 * (x + 1))) for x in range(6)]
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
    # 4) Print scenario content and results.
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

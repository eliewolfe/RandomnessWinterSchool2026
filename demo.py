"""End-to-end demos using direct quantum -> scenario constructors."""

from __future__ import annotations

import numpy as np
import sympy as sp

from randomness_contextuality_lp.contextuality import contextuality_robustness_to_dephasing
from randomness_contextuality_lp.quantum import (
    discover_operational_equivalences_from_gpt_objects,
    projector,
    projector_hs_vector,
    probability_table_from_gpt_vectors,
)
from randomness_contextuality_lp.randomness import eve_optimal_guessing_probability, run_quantum_example
from randomness_contextuality_lp.scenario import ContextualityScenario


def _xz_plane_ket(theta: sp.Expr | float | int) -> sp.Matrix:
    """Real-amplitude qubit ket with Bloch vector in the X-Z plane."""
    theta_sym = sp.sympify(theta)
    return sp.Matrix([sp.cos(theta_sym / 2), sp.sin(theta_sym / 2)])


def _normalize_integer_rays_symbolic(rays: np.ndarray) -> list[sp.Matrix]:
    """Normalize integer-valued ray rows exactly using SymPy."""
    ray_array = np.asarray(rays, dtype=int)
    if ray_array.ndim != 2:
        raise ValueError("rays must have shape (N, d).")

    normalized_kets: list[sp.Matrix] = []
    for row in ray_array:
        row_sym = [sp.Integer(int(entry)) for entry in row]
        norm_sq = sum(entry * entry for entry in row_sym)
        if norm_sq == 0:
            raise ValueError("Cannot normalize zero ray.")
        norm = sp.sqrt(norm_sq)
        normalized_kets.append(sp.Matrix([entry / norm for entry in row_sym]))
    return normalized_kets


def _print_title(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _print_manual_target_randomness(
    scenario: ContextualityScenario,
    measurement_indices: list[tuple[int, ...]],
    target_pair: tuple[int, int],
    bin_outcomes: list[list[int]] | tuple[tuple[int, ...], ...] | None = None,
) -> None:
    x_target, y_target = target_pair
    p_guess_eve = eve_optimal_guessing_probability(
        scenario,
        x=x_target,
        y=y_target,
        bin_outcomes=bin_outcomes,
    )
    p_guess_alice = scenario.alice_optimal_guessing_probability(
        x=x_target,
        y=y_target,
        bin_outcomes=bin_outcomes,
    )
    target_label = "Bob's outcome bins" if bin_outcomes is not None else "Bob's outcome"
    print(f"\nEve optimal guessing probability for {target_label} at target setting:")
    print(f"x_target={x_target}, y_target={y_target}")
    print(f"measurement indices={measurement_indices[y_target]}")
    if bin_outcomes is not None:
        print(f"bins={bin_outcomes}")
    print(f"P_guess = {p_guess_eve:.10f}")
    print(f"\nAlice optimal guessing probability for {target_label} at target setting:")
    print(f"x_target={x_target}, y_target={y_target}")
    if bin_outcomes is not None:
        print(f"bins={bin_outcomes}")
    print(f"P_guess = {p_guess_alice:.10f}")


def _print_manual_target_robustness(scenario: ContextualityScenario, example_label: str) -> None:
    robustness = contextuality_robustness_to_dephasing(scenario)
    print(f"\nContextuality robustness to dephasing ({example_label}):")
    print(f"r* = {robustness:.10f}")
    print("Interpretation: larger r* means more contextual (more dephasing needed to classicalize).")


def _print_measurement_index_sets(measurement_indices: list[tuple[int, ...]]) -> None:
    print("\nProvided measurement index sets (no inference):")
    for y, idx in enumerate(measurement_indices):
        print(f"y={y}: effects {idx}")


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    ket0 = _xz_plane_ket(0)
    ket1 = _xz_plane_ket(sp.pi)
    ket_plus = _xz_plane_ket(sp.pi / 2)
    ket_minus = _xz_plane_ket(-sp.pi / 2)

    ket_pz_plus = _xz_plane_ket(sp.pi / 4)
    ket_pz_minus = _xz_plane_ket(5 * sp.pi / 4)

    ket_mz_plus = _xz_plane_ket(3 * sp.pi / 4)
    ket_mz_minus = _xz_plane_ket(-sp.pi / 4)

    quantum_states = np.array(
        [
            projector(ket0),
            projector(ket1),
            projector(ket_plus),
            projector(ket_minus),
        ],
        dtype=object,
    )  # (X=4, d, d)

    # Example 1: effects +/-Z, +/-X, +/- (X+Z).
    effects_example_1 = np.array(
        [
            projector(ket0),       # +Z
            projector(ket1),       # -Z
            projector(ket_plus),   # +X
            projector(ket_minus),  # -X
            projector(ket_pz_plus),   # +(X+Z)
            projector(ket_pz_minus),  # -(X+Z)
        ],
        dtype=object,
    )
    scenario_1, _, _ = run_quantum_example(
        title="Example 1: Z, X, and (X+Z) measurements",
        quantum_states=quantum_states,
        quantum_effect_set=effects_example_1,
        target_pair=(0, 2),  # target setting (x=0, y=2), i.e. (X+Z)
    )
    _print_manual_target_robustness(scenario_1, "Example 1")

    # Example 2: effects +/- (X+Z), +/- (X-Z).
    effects_example_2 = np.array(
        [
            projector(ket_pz_plus),   # +(X+Z)
            projector(ket_pz_minus),  # -(X+Z)
            projector(ket_mz_plus),   # +(X-Z)
            projector(ket_mz_minus),  # -(X-Z)
        ],
        dtype=object,
    )
    scenario_2, _, _ = run_quantum_example(
        title="Example 2: (X+Z) and (X-Z) measurements",
        quantum_states=quantum_states,
        quantum_effect_set=effects_example_2,
        target_pair=(0, 1),  # target setting (x=0, y=1)
    )
    _print_manual_target_robustness(scenario_2, "Example 2")

    # Example 3: 6 hexagon preparations/effects in the X-Z plane.
    # Preparations are grouped into 3 settings of 2 outcomes (|A|=2).
    thetas = [sp.Integer(k) * sp.pi / 3 for k in range(6)]
    hex_kets = [_xz_plane_ket(theta) for theta in thetas]

    # Pair opposite vertices so each preparation setting averages to maximally mixed.
    prep_pairs = [(0, 3), (1, 4), (2, 5)]
    measurement_indices_3 = [(0, 3), (1, 4), (2, 5)]
    gpt_states_3 = np.array(
        [
            [projector_hs_vector(hex_kets[i]), projector_hs_vector(hex_kets[j])]
            for (i, j) in prep_pairs
        ],
        dtype=object,
    )
    gpt_effects_3 = np.array(
        [
            [projector_hs_vector(hex_kets[i]), projector_hs_vector(hex_kets[j])]
            for (i, j) in measurement_indices_3
        ],
        dtype=object,
    )
    opeq_preps_3 = discover_operational_equivalences_from_gpt_objects(gpt_states_3)
    opeq_meas_3 = discover_operational_equivalences_from_gpt_objects(gpt_effects_3)
    data_3 = probability_table_from_gpt_vectors(gpt_states_3, gpt_effects_3)
    _print_title("Example 3: Hexagon states/effects in X-Z plane (|A|=2)")
    scenario_3 = ContextualityScenario(
        data=np.real_if_close(data_3).astype(float),
        opeq_preps=np.real_if_close(opeq_preps_3).astype(float),
        opeq_meas=np.real_if_close(opeq_meas_3).astype(float),
        verbose=True,
    )
    _print_measurement_index_sets(measurement_indices_3)
    _print_manual_target_randomness(scenario_3, measurement_indices_3, target_pair=(0, 0))
    _print_manual_target_robustness(scenario_3, "Example 3")

    # Example 4: Cabello-style 18-ray KS set in d=4, using GPT constructor directly.
    # 9 measurements, each with 4 outcomes; each effect appears in multiple measurements.
    labels = list("123456789ABCDEFGHI")
    cabello_rays = np.array(
        [
            [1, 0, 0, 0],   # 1
            [0, 1, 0, 0],   # 2
            [0, 0, 1, 0],   # 3
            [1, 1, 1, 1],   # 4
            [1, -1, 1, -1], # 5
            [1, -1, -1, 1], # 6
            [1, -1, -1, -1],# 7
            [1, -1, 1, 1],  # 8
            [1, 1, 1, -1],  # 9
            [1, 1, 0, 0],   # A
            [0, 0, 1, 1],   # B
            [0, 0, 1, -1],  # C
            [0, 1, 0, 1],   # D
            [0, 1, 0, -1],  # E
            [1, 0, -1, 0],  # F
            [1, 0, 0, -1],  # G
            [1, 0, 0, 1],   # H
            [0, 1, -1, 0],  # I
        ],
        dtype=int,
    )
    cabello_kets = _normalize_integer_rays_symbolic(cabello_rays)
    label_to_index = {lab: i for i, lab in enumerate(labels)}

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
    context_indices_4 = [tuple(label_to_index[ch] for ch in context) for context in contexts]

    gpt_effect_set_example_4 = np.array([projector_hs_vector(ket) for ket in cabello_kets], dtype=object)
    gpt_effects_grouped_example_4 = np.array(
        [[gpt_effect_set_example_4[idx] for idx in context] for context in context_indices_4],
        dtype=object,
    )  # (Y=9, B=4, K)
    gpt_states_example_4 = np.array(
        [[projector_hs_vector(cabello_kets[idx]) for idx in context] for context in context_indices_4],
        dtype=object,
    )
    opeq_preps_4 = discover_operational_equivalences_from_gpt_objects(gpt_states_example_4)
    opeq_meas_4 = discover_operational_equivalences_from_gpt_objects(gpt_effects_grouped_example_4)
    data_4 = probability_table_from_gpt_vectors(gpt_states_example_4, gpt_effects_grouped_example_4)
    _print_title("Example 4: Cabello 18-ray KS set via GPT constructor (9 x 4 contexts)")
    scenario_4 = ContextualityScenario(
        data=np.real_if_close(data_4).astype(float),
        opeq_preps=np.real_if_close(opeq_preps_4).astype(float),
        opeq_meas=np.real_if_close(opeq_meas_4).astype(float),
        verbose=False,
    )
    print(scenario_4)
    # print("\nData table P(a,b|x,y):")
    # scenario_4.print_probabilities(as_p_b_given_x_y=False)
    _print_measurement_index_sets(context_indices_4)
    _print_manual_target_randomness(scenario_4, context_indices_4, target_pair=(0, 0))
    _print_manual_target_robustness(scenario_4, "Example 4")

    # Example 5: Peres 24-ray construction restricted to 6 disjoint bases.
    # Grouping is by contiguous 4-ray blocks from the screenshot:
    # {1,2,3,4}, {5,6,7,8}, ..., {21,22,23,24}.

    # Peres 24 rays from the screenshot (barred digit -> negative entry).
    peres_rays = np.array(
        [
            [2, 0, 0, 0],   # 1
            [0, 2, 0, 0],   # 2
            [0, 0, 2, 0],   # 3
            [0, 0, 0, 2],   # 4
            [1, 1, 1, 1],   # 5
            [1, 1, -1, -1], # 6
            [1, -1, 1, -1], # 7
            [1, -1, -1, 1], # 8
            [1, -1, -1, -1],# 9
            [1, -1, 1, 1],  # 10
            [1, 1, -1, 1],  # 11
            [1, 1, 1, -1],  # 12
            [1, 1, 0, 0],   # 13
            [1, -1, 0, 0],  # 14
            [0, 0, 1, 1],   # 15
            [0, 0, 1, -1],  # 16
            [0, 1, 0, 1],   # 17
            [0, 1, 0, -1],  # 18
            [1, 0, 1, 0],   # 19
            [1, 0, -1, 0],  # 20
            [1, 0, 0, -1],  # 21
            [1, 0, 0, 1],   # 22
            [0, 1, -1, 0],  # 23
            [0, 1, 1, 0],   # 24
        ],
        dtype=int,
    )
    peres_kets = _normalize_integer_rays_symbolic(peres_rays)

    context_indices_5 = [tuple(range(4 * y, 4 * (y + 1))) for y in range(6)]
    gpt_effect_set_example_5 = np.array([projector_hs_vector(ket) for ket in peres_kets], dtype=object)
    gpt_effects_grouped_example_5 = np.array(
        [[gpt_effect_set_example_5[idx] for idx in context] for context in context_indices_5],
        dtype=object,
    )  # (6, 4, K)
    gpt_states_example_5 = np.array(gpt_effects_grouped_example_5, copy=True)

    opeq_preps_5 = discover_operational_equivalences_from_gpt_objects(gpt_states_example_5)
    opeq_meas_5 = discover_operational_equivalences_from_gpt_objects(gpt_effects_grouped_example_5)
    data_5 = probability_table_from_gpt_vectors(gpt_states_example_5, gpt_effects_grouped_example_5)

    _print_title("Example 5: Peres 24 rays in 6 disjoint 4-ray bases")
    scenario_5 = ContextualityScenario(
        data=np.real_if_close(data_5).astype(float),
        opeq_preps=np.real_if_close(opeq_preps_5).astype(float),
        opeq_meas=np.real_if_close(opeq_meas_5).astype(float),
        verbose=False,
    )
    print(scenario_5)
    _print_measurement_index_sets(context_indices_5)
    _print_manual_target_randomness(
        scenario_5,
        context_indices_5,
        target_pair=(0, 0),
    )
    _print_manual_target_robustness(scenario_5, "Example 5")


if __name__ == "__main__":
    main()

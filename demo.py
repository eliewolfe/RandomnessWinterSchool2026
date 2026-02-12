"""End-to-end demos using direct quantum -> scenario constructors."""

from __future__ import annotations

import numpy as np

from randomness_contextuality_lp.contextuality import contextuality_robustness_to_dephasing
from randomness_contextuality_lp.quantum import (
    discover_operational_equivalences_from_gpt_objects,
    probability_table_from_gpt_vectors,
    projector,
    projector_hs_vector,
)
from randomness_contextuality_lp.randomness import eve_optimal_guessing_probability, run_quantum_example
from randomness_contextuality_lp.scenario import ContextualityScenario


def _xz_plane_ket(theta: float) -> np.ndarray:
    """Real-amplitude qubit ket with Bloch vector in the X-Z plane."""
    return np.array([np.cos(theta / 2.0), np.sin(theta / 2.0)], dtype=complex)


def _print_title(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _print_manual_target_randomness(
    scenario: ContextualityScenario,
    measurement_indices: list[tuple[int, ...]],
    target_pair: tuple[int, int],
) -> None:
    x_target, y_target = target_pair
    p_guess_eve = eve_optimal_guessing_probability(scenario, x=x_target, y=y_target)
    p_guess_alice = scenario.alice_optimal_guessing_probability(x=x_target, y=y_target)
    print("\nEve optimal guessing probability for target setting:")
    print(f"x_target={x_target}, y_target={y_target}")
    print(f"measurement indices={measurement_indices[y_target]}")
    print(f"P_guess = {p_guess_eve:.10f}")
    print("\nAlice optimal guessing probability for target setting:")
    print(f"x_target={x_target}, y_target={y_target}")
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

    ket0 = np.array([1.0, 0.0], dtype=complex)
    ket1 = np.array([0.0, 1.0], dtype=complex)
    ket_plus = (ket0 + ket1) / np.sqrt(2.0)
    ket_minus = (ket0 - ket1) / np.sqrt(2.0)

    quantum_states = np.array(
        [projector(ket0), projector(ket1), projector(ket_plus), projector(ket_minus)],
        dtype=complex,
    )  # (X=4, d, d)

    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    op_x_plus_z = (sigma_x + sigma_z) / np.sqrt(2.0)
    eigvals_pz, eigvecs_pz = np.linalg.eigh(op_x_plus_z)
    ket_pz_minus = eigvecs_pz[:, np.argmin(eigvals_pz)]
    ket_pz_plus = eigvecs_pz[:, np.argmax(eigvals_pz)]

    op_x_minus_z = (sigma_x - sigma_z) / np.sqrt(2.0)
    eigvals_mz, eigvecs_mz = np.linalg.eigh(op_x_minus_z)
    ket_mz_minus = eigvecs_mz[:, np.argmin(eigvals_mz)]
    ket_mz_plus = eigvecs_mz[:, np.argmax(eigvals_mz)]

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
        dtype=complex,
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
        dtype=complex,
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
    thetas = np.arange(6, dtype=float) * (np.pi / 3.0)
    hex_kets = [_xz_plane_ket(theta) for theta in thetas]

    # Pair opposite vertices so each preparation setting averages to maximally mixed.
    prep_pairs = [(0, 3), (1, 4), (2, 5)]
    measurement_indices_3 = [(0, 3), (1, 4), (2, 5)]
    gpt_states_3 = np.array(
        [[projector_hs_vector(hex_kets[i]), projector_hs_vector(hex_kets[j])] for (i, j) in prep_pairs],
        dtype=complex,
    )
    gpt_effects_3 = np.array(
        [[projector_hs_vector(hex_kets[i]), projector_hs_vector(hex_kets[j])] for (i, j) in measurement_indices_3],
        dtype=complex,
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
    ray_coords = np.array(
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
        dtype=float,
    )
    ray_coords = ray_coords / np.linalg.norm(ray_coords, axis=1, keepdims=True)
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
    context_indices = [tuple(label_to_index[ch] for ch in context) for context in contexts]

    kets_4d = ray_coords.astype(complex)
    gpt_effect_set_example_4 = np.array([projector_hs_vector(ket) for ket in kets_4d], dtype=complex)
    gpt_effects_grouped_example_4 = np.array(
        [[gpt_effect_set_example_4[idx] for idx in context] for context in context_indices],
        dtype=complex,
    )  # (Y=9, B=4, K)
    gpt_states_example_4 = np.array(
        [[projector_hs_vector(kets_4d[idx]) for idx in context] for context in context_indices],
        dtype=complex,
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
    _print_measurement_index_sets(context_indices)
    _print_manual_target_randomness(scenario_4, context_indices, target_pair=(0, 0))
    _print_manual_target_robustness(scenario_4, "Example 4")


if __name__ == "__main__":
    main()

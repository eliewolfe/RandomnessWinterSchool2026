"""QKD-focused end-to-end demos with clustered preparations in all examples."""

from __future__ import annotations

import numpy as np
import sympy as sp

from randomness_contextuality_lp.contextuality import contextuality_robustness_to_dephasing
from randomness_contextuality_lp.quantum import (
    discover_operational_equivalences_from_gpt_objects,
    projector_hs_vector,
    probability_table_from_gpt_vectors,
)
from randomness_contextuality_lp.randomness import analyze_scenario
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


def _format_decimal(value: float, decimals: int = 3) -> str:
    rounded = round(float(value), decimals)
    if abs(rounded) < 10 ** (-decimals):
        rounded = 0.0
    text = f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")
    if text in {"-0", ""}:
        return "0"
    return text


def _print_guessing_probability_grids(
    scenario: ContextualityScenario,
    measurement_indices: list[tuple[int, ...]],
    precision: int = 3,
    include_keyrate_pairs: bool = True,
    keyrate_threshold: float = 0.1,
    guess_who: str = "Bob",
) -> None:
    target = scenario._normalize_guess_who(guess_who)
    p_guess_eve, keyrate_table = analyze_scenario(scenario=scenario, guess_who=target)
    num_x = scenario.X_cardinality
    num_y = scenario.Y_cardinality

    if target == "Bob":
        p_guess_native = scenario.alice_optimal_guessing_bob_probability
        target_label = "Bob's outcome"
        native_label = "Alice optimal"
    elif target == "Alice":
        p_guess_native = scenario.bob_optimal_guessing_alice
        target_label = "Alice's outcome"
        native_label = "Bob optimal"
    else:
        p_guess_native = scenario.largest_joint_probability
        target_label = "the joint (Alice, Bob) outcome pair"
        native_label = "Largest joint"

    float_formatter = {"float_kind": lambda value: _format_decimal(value, decimals=precision)}
    print(f"\nEve optimal guessing probabilities for {target_label} (rows: x, columns: y):")
    print(np.array2string(p_guess_eve, formatter=float_formatter))
    print(f"\n{native_label} guessing probabilities for {target_label} (rows: x, columns: y):")
    print(np.array2string(p_guess_native, formatter=float_formatter))
    if include_keyrate_pairs:
        qualifying_pairs = [
            (x, y, keyrate_table[x, y])
            for x, y in np.ndindex(num_x, num_y)
            if keyrate_table[x, y] > keyrate_threshold
        ]
        if target == "Bob":
            print("\nTaking Bob's outcomes as the master key, then the strictly positive key rate pairings are:")
        elif target == "Alice":
            print("\nTaking Alice's outcomes as the master key, then the strictly positive key rate pairings are:")
        else:
            print("\nTaking joint (Alice, Bob) outcomes as the master key, then the strictly positive key rate pairings are:")
        if not qualifying_pairs:
            print("none")
        else:
            for x, y, value in qualifying_pairs:
                print(
                    f"(x={x}, y={y}) -> "
                    f"{_format_decimal(float(value), decimals=precision)}"
                )


def _print_manual_target_robustness(scenario: ContextualityScenario, example_label: str) -> None:
    robustness = contextuality_robustness_to_dephasing(scenario)
    print(f"\nContextuality robustness to dephasing ({example_label}):")
    print(f"r* = {_format_decimal(robustness, decimals=3)}")
    print("Interpretation: larger r* means more contextual (more dephasing needed to classicalize).")


def _print_measurement_index_sets(measurement_indices: list[tuple[int, ...]]) -> None:
    print("\nProvided measurement index sets:")
    for y, idx in enumerate(measurement_indices):
        print(f"y={y}: effects {idx}")


def _print_preparation_index_sets(preparation_indices: list[tuple[int, ...]]) -> None:
    print("\nProvided preparation index sets:")
    for x, idx in enumerate(preparation_indices):
        print(f"x={x}: preparations {idx}")


def _print_measurement_operational_equivalences(
    scenario: ContextualityScenario,
    precision: int = 3,
) -> None:
    print()
    scenario.print_measurement_operational_equivalences(
        precision=precision,
        representation="symbolic",
    )


def _group_gpt_vectors(
    gpt_vector_set: np.ndarray,
    grouped_indices: list[tuple[int, ...]],
) -> np.ndarray:
    return np.array(
        [[gpt_vector_set[idx] for idx in index_group] for index_group in grouped_indices],
        dtype=object,
    )


def _build_manual_scenario_from_grouped_gpt(
    gpt_states_grouped: np.ndarray,
    gpt_effects_grouped: np.ndarray,
    verbose: bool = False,
) -> ContextualityScenario:
    data_symbolic = probability_table_from_gpt_vectors(gpt_states_grouped, gpt_effects_grouped)
    opeq_preps_symbolic = discover_operational_equivalences_from_gpt_objects(gpt_states_grouped)
    opeq_meas_symbolic = discover_operational_equivalences_from_gpt_objects(gpt_effects_grouped)
    return ContextualityScenario(
        data=data_symbolic,
        opeq_preps=opeq_preps_symbolic,
        opeq_meas=opeq_meas_symbolic,
        verbose=verbose,
    )


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

    # Example 1: +/-Z, +/-X, and +/- (X+Z), with antipodal preparation clustering.
    _print_title("Example 1: Z, X, and (X+Z) measurements")
    state_kets_example_1 = [ket0, ket1, ket_plus, ket_minus]
    effect_kets_example_1 = [ket0, ket1, ket_plus, ket_minus, ket_pz_plus, ket_pz_minus]
    preparation_indices_example_1 = [(0, 1), (2, 3)]
    measurement_indices_example_1 = [(0, 1), (2, 3), (4, 5)]

    gpt_state_set_example_1 = np.array([projector_hs_vector(ket) for ket in state_kets_example_1], dtype=object)
    gpt_effect_set_example_1 = np.array([projector_hs_vector(ket) for ket in effect_kets_example_1], dtype=object)
    gpt_states_grouped_example_1 = _group_gpt_vectors(
        gpt_state_set_example_1,
        preparation_indices_example_1,
    )
    gpt_effects_grouped_example_1 = _group_gpt_vectors(
        gpt_effect_set_example_1,
        measurement_indices_example_1,
    )
    scenario_example_1 = _build_manual_scenario_from_grouped_gpt(
        gpt_states_grouped_example_1,
        gpt_effects_grouped_example_1,
        verbose=False,
    )
    _print_preparation_index_sets(preparation_indices_example_1)
    _print_measurement_index_sets(measurement_indices_example_1)
    _print_measurement_operational_equivalences(scenario_example_1)
    print("\nSymbolic probability table P(a,b|x,y):")
    scenario_example_1.print_probabilities(
        as_p_b_given_x_y=False,
        precision=3,
        representation="symbolic",
    )
    _print_guessing_probability_grids(
        scenario_example_1,
        measurement_indices_example_1,
        guess_who="Bob",
    )
    _print_manual_target_robustness(scenario_example_1, "Example 1")

    # Example 2: +/- (X+Z), +/- (X-Z), with antipodal preparation clustering.
    _print_title("Example 2: (X+Z) and (X-Z) measurements")
    state_kets_example_2 = [ket0, ket1, ket_plus, ket_minus]
    effect_kets_example_2 = [ket_pz_plus, ket_pz_minus, ket_mz_plus, ket_mz_minus]
    preparation_indices_example_2 = [(0, 1), (2, 3)]
    measurement_indices_example_2 = [(0, 1), (2, 3)]

    gpt_state_set_example_2 = np.array([projector_hs_vector(ket) for ket in state_kets_example_2], dtype=object)
    gpt_effect_set_example_2 = np.array([projector_hs_vector(ket) for ket in effect_kets_example_2], dtype=object)
    gpt_states_grouped_example_2 = _group_gpt_vectors(
        gpt_state_set_example_2,
        preparation_indices_example_2,
    )
    gpt_effects_grouped_example_2 = _group_gpt_vectors(
        gpt_effect_set_example_2,
        measurement_indices_example_2,
    )
    scenario_example_2 = _build_manual_scenario_from_grouped_gpt(
        gpt_states_grouped_example_2,
        gpt_effects_grouped_example_2,
        verbose=False,
    )
    _print_preparation_index_sets(preparation_indices_example_2)
    _print_measurement_index_sets(measurement_indices_example_2)
    _print_measurement_operational_equivalences(scenario_example_2)
    print("\nSymbolic probability table P(a,b|x,y):")
    scenario_example_2.print_probabilities(
        as_p_b_given_x_y=False,
        precision=3,
        representation="symbolic",
    )
    _print_guessing_probability_grids(
        scenario_example_2,
        measurement_indices_example_2,
        guess_who="Bob",
    )
    _print_manual_target_robustness(scenario_example_2, "Example 2")

    # Example 3: 6 hexagon preparations/effects in the X-Z plane, grouped into 3 settings of 2 outcomes.
    thetas = [sp.Integer(k) * sp.pi / 3 for k in range(6)]
    state_kets_example_3 = [_xz_plane_ket(theta) for theta in thetas]
    effect_kets_example_3 = list(state_kets_example_3)

    # Pair opposite vertices so each preparation setting averages to maximally mixed.
    preparation_indices_example_3 = [(0, 3), (1, 4), (2, 5)]
    measurement_indices_example_3 = [(0, 3), (1, 4), (2, 5)]

    gpt_state_set_example_3 = np.array([projector_hs_vector(ket) for ket in state_kets_example_3], dtype=object)
    gpt_effect_set_example_3 = np.array([projector_hs_vector(ket) for ket in effect_kets_example_3], dtype=object)
    gpt_states_grouped_example_3 = _group_gpt_vectors(
        gpt_state_set_example_3,
        preparation_indices_example_3,
    )
    gpt_effects_grouped_example_3 = _group_gpt_vectors(
        gpt_effect_set_example_3,
        measurement_indices_example_3,
    )
    _print_title("Example 3: Hexagon states/effects in X-Z plane (|A|=2)")
    scenario_example_3 = _build_manual_scenario_from_grouped_gpt(
        gpt_states_grouped_example_3,
        gpt_effects_grouped_example_3,
        verbose=False,
    )
    _print_preparation_index_sets(preparation_indices_example_3)
    _print_measurement_index_sets(measurement_indices_example_3)
    _print_measurement_operational_equivalences(scenario_example_3)
    print("\nSymbolic probability table P(a,b|x,y):")
    scenario_example_3.print_probabilities(
        as_p_b_given_x_y=False,
        precision=3,
        representation="symbolic",
    )
    _print_guessing_probability_grids(
        scenario_example_3,
        measurement_indices_example_3,
        guess_who="Bob",
    )
    _print_manual_target_robustness(scenario_example_3, "Example 3")

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
    preparation_indices_example_4 = [tuple(label_to_index[ch] for ch in context) for context in contexts]
    measurement_indices_example_4 = list(preparation_indices_example_4)

    gpt_state_set_example_4 = np.array([projector_hs_vector(ket) for ket in cabello_kets], dtype=object)
    gpt_effect_set_example_4 = np.array([projector_hs_vector(ket) for ket in cabello_kets], dtype=object)
    gpt_states_grouped_example_4 = _group_gpt_vectors(
        gpt_state_set_example_4,
        preparation_indices_example_4,
    )
    gpt_effects_grouped_example_4 = _group_gpt_vectors(
        gpt_effect_set_example_4,
        measurement_indices_example_4,
    )

    _print_title("Example 4: Cabello 18-ray KS set via GPT constructor (9 x 4 contexts)")
    scenario_example_4 = _build_manual_scenario_from_grouped_gpt(
        gpt_states_grouped_example_4,
        gpt_effects_grouped_example_4,
        verbose=False,
    )
    print(scenario_example_4)
    # print("\nData table P(a,b|x,y):")
    # scenario_4.print_probabilities(as_p_b_given_x_y=False)
    _print_preparation_index_sets(preparation_indices_example_4)
    _print_measurement_index_sets(measurement_indices_example_4)
    _print_measurement_operational_equivalences(scenario_example_4)
    _print_guessing_probability_grids(
        scenario_example_4,
        measurement_indices_example_4,
        guess_who="Bob",
    )
    _print_manual_target_robustness(scenario_example_4, "Example 4")

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

    preparation_indices_example_5 = [tuple(range(4 * x, 4 * (x + 1))) for x in range(6)]
    measurement_indices_example_5 = list(preparation_indices_example_5)

    gpt_state_set_example_5 = np.array([projector_hs_vector(ket) for ket in peres_kets], dtype=object)
    gpt_effect_set_example_5 = np.array([projector_hs_vector(ket) for ket in peres_kets], dtype=object)
    gpt_states_grouped_example_5 = _group_gpt_vectors(
        gpt_state_set_example_5,
        preparation_indices_example_5,
    )
    gpt_effects_grouped_example_5 = _group_gpt_vectors(
        gpt_effect_set_example_5,
        measurement_indices_example_5,
    )

    _print_title("Example 5: Peres 24 rays in 6 disjoint 4-ray bases")
    scenario_example_5 = _build_manual_scenario_from_grouped_gpt(
        gpt_states_grouped_example_5,
        gpt_effects_grouped_example_5,
        verbose=False,
    )
    print(scenario_example_5)
    _print_preparation_index_sets(preparation_indices_example_5)
    _print_measurement_index_sets(measurement_indices_example_5)
    _print_measurement_operational_equivalences(scenario_example_5)
    _print_guessing_probability_grids(
        scenario_example_5,
        measurement_indices_example_5,
        guess_who="Bob",
    )
    _print_manual_target_robustness(scenario_example_5, "Example 5")


if __name__ == "__main__":
    main()

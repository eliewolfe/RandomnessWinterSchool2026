"""Pedagogical QKD demo: (3,2)-PORAC as one-setting/eight-outcome source.

Recommended execution:
    python -m contextualityqkd.demos.qkd_porac_3_2
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
    QuantumContextualityScenario,
)
from contextualityqkd.scenario import ContextualityScenario


def _porac_index(x0: int, x1: int, x2: int) -> int:
    """Encode bit triple (x0,x1,x2) as integer x in {0,...,7}."""
    return int(4 * x0 + 2 * x1 + x2)


def porac_bit(a: int, y: int) -> int:
    """Return a_y (the y-th bit of Alice outcome label a in {0,...,7})."""
    if y < 0 or y > 2:
        raise ValueError("y must be 0, 1, or 2.")
    return int((int(a) >> (2 - int(y))) & 1)


def porac_objective_label_map(x: int, y: int, a: int, b: int) -> int:
    """Map each event to Eve's target label e=a_y for PORAC."""
    _ = x
    _ = b
    return porac_bit(a, y)


def _porac_article_prep_opeq_rows() -> sp.Matrix:
    """Return the seven reference PORAC preparation-constraint rows from the article form."""
    rows: list[list[sp.Rational]] = []
    for x2 in (0, 1):
        coeffs = [sp.Integer(0)] * 8
        coeffs[_porac_index(0, 0, x2)] += sp.Rational(1, 2)
        coeffs[_porac_index(1, 1, x2)] += sp.Rational(1, 2)
        coeffs[_porac_index(0, 1, x2)] -= sp.Rational(1, 2)
        coeffs[_porac_index(1, 0, x2)] -= sp.Rational(1, 2)
        rows.append(coeffs)
    for x1 in (0, 1):
        coeffs = [sp.Integer(0)] * 8
        coeffs[_porac_index(0, x1, 0)] += sp.Rational(1, 2)
        coeffs[_porac_index(1, x1, 1)] += sp.Rational(1, 2)
        coeffs[_porac_index(0, x1, 1)] -= sp.Rational(1, 2)
        coeffs[_porac_index(1, x1, 0)] -= sp.Rational(1, 2)
        rows.append(coeffs)
    for x0 in (0, 1):
        coeffs = [sp.Integer(0)] * 8
        coeffs[_porac_index(x0, 0, 0)] += sp.Rational(1, 2)
        coeffs[_porac_index(x0, 1, 1)] += sp.Rational(1, 2)
        coeffs[_porac_index(x0, 0, 1)] -= sp.Rational(1, 2)
        coeffs[_porac_index(x0, 1, 0)] -= sp.Rational(1, 2)
        rows.append(coeffs)
    coeffs = [sp.Integer(0)] * 8
    for bits in ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)):
        coeffs[_porac_index(*bits)] += sp.Rational(1, 4)
    for bits in ((0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)):
        coeffs[_porac_index(*bits)] -= sp.Rational(1, 4)
    rows.append(coeffs)
    return sp.Matrix(rows)


def _print_porac_article_prep_opeqs() -> None:
    """Print the article preparation OPEQ rows in scenario-style ragged format."""
    rows = _porac_article_prep_opeq_rows()
    print(f"\nPreparation OPEQs according to the article ({rows.rows} rows):")
    for k in range(rows.rows):
        row_entries = [ContextualityScenario._format_symbolic_entry(rows[k, a], precision=6) for a in range(rows.cols)]
        ragged = "[[" + ", ".join(row_entries) + "]]"
        print(f"k={k}: {ragged}")


def _validate_porac_prep_opeq_subspace(scenario: ContextualityScenario) -> tuple[int, int, int]:
    """Validate that discovered prep OPEQs match article PORAC constraints over outcome labels."""
    article = _porac_article_prep_opeq_rows()
    if scenario.X_cardinality != 1:
        raise ValueError("Expected X_cardinality=1 in one-setting PORAC demo.")
    discovered = sp.Matrix(np.asarray(scenario.opeq_preps_symbolic, dtype=object).reshape(-1, scenario.A_cardinality))
    rank_article = int(article.rank())
    rank_discovered = int(discovered.rank())
    rank_stacked = int(sp.Matrix.vstack(article, discovered).rank())
    if not (rank_article == rank_discovered == rank_stacked):
        raise ValueError(
            "Auto-discovered preparation OPEQs are not equivalent to PORAC article constraints: "
            f"rank(article)={rank_article}, rank(discovered)={rank_discovered}, "
            f"rank(vstack)={rank_stacked}."
        )
    return rank_article, rank_discovered, rank_stacked


def build_porac_scenario(*, eta: float = 1.0) -> QuantumContextualityScenario:
    """Construct one-setting/outcome-lifted (3,2)-PORAC from a canonical 3->1 RAC strategy.

    Model summary:
    - 1 preparation setting (x=0) with 8 outcomes (a in {0,...,7})
      interpreted as bit triples a0a1a2
    - 3 binary measurements (y in {0,1,2})
    - Bob output b in {0,1}
    - parity-oblivious preparation OPEQs induced across Alice outcomes
    """
    eta_f = float(eta)
    if eta_f < 0.0 or eta_f > 1.0:
        raise ValueError("eta must lie in [0,1].")

    # Pauli basis and identity for Bloch-vector state construction.
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    paulis = [sigma_x, sigma_y, sigma_z]

    # ---------------------------------------------------------------------
    # 1) Build 8 qubit states at cube corners:
    #      rho_a = (I + r_a . sigma)/2
    #    then apply optional isotropic depolarizing noise parameter eta.
    #    Group all eight as outcomes of a single Alice setting x=0.
    # ---------------------------------------------------------------------
    single_setting_outcomes: list[np.ndarray] = []
    for a0 in (0, 1):
        for a1 in (0, 1):
            for a2 in (0, 1):
                r = np.array([(-1) ** a0, (-1) ** a1, (-1) ** a2], dtype=float) / np.sqrt(3.0)
                rho = 0.5 * (identity + r[0] * sigma_x + r[1] * sigma_y + r[2] * sigma_z)
                rho_eta = eta_f * rho + (1.0 - eta_f) * 0.5 * identity
                single_setting_outcomes.append(rho_eta)
    quantum_states_grouped = [single_setting_outcomes]

    # ---------------------------------------------------------------------
    # 2) Bob's three binary measurements are Pauli X, Y, Z projective effects.
    # ---------------------------------------------------------------------
    quantum_effects_grouped: list[list[np.ndarray]] = []
    for y in range(3):
        plus = 0.5 * (identity + paulis[y])
        minus = 0.5 * (identity - paulis[y])
        quantum_effects_grouped.append([plus, minus])

    # ---------------------------------------------------------------------
    # 3) Instantiate scenario directly from quantum primitives.
    # ---------------------------------------------------------------------
    return QuantumContextualityScenario(
        quantum_states=np.asarray(quantum_states_grouped, dtype=complex),
        quantum_effects=np.asarray(quantum_effects_grouped, dtype=complex),
        verbose=False,
    )


def porac_bob_success_probability(scenario: ContextualityScenario) -> float:
    """Compute S_B^rac = average_y sum_a P(a,b=a_y|x=0,y)."""
    if scenario.X_cardinality != 1:
        raise ValueError("Expected X_cardinality=1 in one-setting PORAC demo.")
    total = 0.0
    for y in range(scenario.Y_cardinality):
        for a in range(scenario.A_cardinality):
            total += float(scenario.data_numeric[0, y, a, porac_bit(a, y)])
    return float(total / scenario.Y_cardinality)


def porac_keyrate_lower_bound(
    scenario: ContextualityScenario,
) -> tuple[float, float, float]:
    """Return (h_b, h_e, r_lb) with h_b=h2(S_B^rac), h_e=-log2(S_E^rac), r_lb=h_e-h_b."""
    s_b_prob = porac_bob_success_probability(scenario)
    s_e_prob = scenario.compute_eve_average_guessing_probability(
        guess_who="Alice",
        objective_label_map=porac_objective_label_map,
    )
    h_b = float(ContextualityScenario.binary_entropy(s_b_prob))
    h_e = float(-np.log2(s_e_prob))
    r_lb = float(h_e - h_b)
    return float(h_b), float(h_e), float(r_lb)


def porac_native_bob_guessing_table_for_bit(scenario: ContextualityScenario) -> np.ndarray:
    """Return Bob-optimal native table for guessing target bit a_y (rows: x, cols: y)."""
    table = np.zeros((scenario.X_cardinality, scenario.Y_cardinality), dtype=float)
    for x in range(scenario.X_cardinality):
        a_count = int(scenario.a_cardinality_per_x[x])
        for y in range(scenario.Y_cardinality):
            b_count = int(scenario.b_cardinality_per_y[y])
            p_guess = 0.0
            for b in range(b_count):
                masses = np.zeros(2, dtype=float)
                for a in range(a_count):
                    label = porac_objective_label_map(x, y, a, b)
                    masses[int(label)] += float(scenario.data_numeric[x, y, a, b])
                p_guess += float(np.max(masses))
            table[x, y] = p_guess
    return table


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    # ---------------------------------------------------------------------
    # Build the noiseless PORAC scenario.
    # ---------------------------------------------------------------------
    scenario = build_porac_scenario(eta=1.0)
    ContextualityScenario.print_title("QKD: (3,2)-PORAC protocol, one-setting eight-outcome form (eta=1.000)")
    scenario.print_preparation_index_sets([tuple(range(8))])

    print("\nSymbolic probability table P(a,b|x,y):")
    scenario.print_probabilities(as_p_b_given_x_y=False, precision=3, representation="symbolic")

    print("\nOperational equivalences:")
    scenario.print_operational_equivalences(precision=3, representation="symbolic")
    _print_porac_article_prep_opeqs()
    rank_article, rank_discovered, rank_stacked = _validate_porac_prep_opeq_subspace(scenario)
    print(
        "\nValidated PORAC prep-OPEQ subspace equivalence to article constraints: "
        f"rank(article)={rank_article}, rank(discovered)={rank_discovered}, rank(vstack)={rank_stacked}."
    )

    print("\nPORAC Eve-target guessing objective (e = a_y, where a is a 3-bit label):")
    eve_bit_table = scenario.compute_eve_guessing_table(
        guess_who="Alice",
        objective_label_map=porac_objective_label_map,
    )
    print("Eve optimal guessing table for target bit a_y (rows: x, columns: y):")
    print(
        np.array2string(
            eve_bit_table,
            formatter={
                "float_kind": lambda value: ContextualityScenario.format_numeric(value, precision=3)
            },
        )
    )
    print("\nNative Bob-optimal guessing table for target bit a_y (rows: x, columns: y):")
    native_bit_table = porac_native_bob_guessing_table_for_bit(scenario)
    print(
        np.array2string(
            native_bit_table,
            formatter={
                "float_kind": lambda value: ContextualityScenario.format_numeric(value, precision=3)
            },
        )
    )

    h_b, h_e, r_lb = porac_keyrate_lower_bound(scenario)
    print("\nPORAC key-rate components (Eq. 22):")
    print(f"h_b = h2(S_B^rac) = {ContextualityScenario.format_numeric(h_b, precision=6)}")
    print(f"h_e = -log2(S_E^rac) = {ContextualityScenario.format_numeric(h_e, precision=6)}")
    print(
        "r_lb = h_e - h_b = "
        f"{ContextualityScenario.format_numeric(r_lb, precision=6)}"
    )


if __name__ == "__main__":
    main()

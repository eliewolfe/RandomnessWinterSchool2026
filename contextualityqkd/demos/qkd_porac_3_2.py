"""Pedagogical QKD demo: (3,2)-PORAC with custom Eve objective mapping.

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


def porac_bit(x: int, y: int) -> int:
    """Return x_y (the y-th retained source bit for setting x)."""
    if y < 0 or y > 2:
        raise ValueError("y must be 0, 1, or 2.")
    return int((int(x) >> (2 - int(y))) & 1)


def porac_objective_label_map(x: int, y: int, a: int, b: int) -> int:
    """Map each event to Eve's target label e=x_y for PORAC."""
    _ = a
    _ = b
    return porac_bit(x, y)


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
        row_entries = [ContextualityScenario._format_symbolic_entry(rows[k, x], precision=6) for x in range(rows.cols)]
        ragged = "[" + ", ".join(f"[{entry}]" for entry in row_entries) + "]"
        print(f"k={k}: {ragged}")


def _validate_porac_prep_opeq_subspace(scenario: ContextualityScenario) -> tuple[int, int, int]:
    """Validate that discovered prep OPEQs span the same row-space as article PORAC constraints."""
    article = _porac_article_prep_opeq_rows()
    discovered = sp.Matrix(np.asarray(scenario.opeq_preps_symbolic, dtype=object).reshape(-1, scenario.X_cardinality))
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
    """Construct the (3,2)-PORAC scenario using a canonical 3->1 RAC strategy.

    Model summary:
    - 8 preparations (x in {0,...,7}, interpreted as bit triples x0x1x2)
    - 3 binary measurements (y in {0,1,2})
    - source outcome cardinality A=1 (singleton preparations)
    - Bob output b in {0,1}
    - parity-oblivious preparation OPEQs explicitly imposed
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
    #      rho_x = (I + r_x . sigma)/2
    #    then apply optional isotropic depolarizing noise parameter eta.
    # ---------------------------------------------------------------------
    quantum_states_grouped: list[list[np.ndarray]] = []
    for x0 in (0, 1):
        for x1 in (0, 1):
            for x2 in (0, 1):
                r = np.array([(-1) ** x0, (-1) ** x1, (-1) ** x2], dtype=float) / np.sqrt(3.0)
                rho = 0.5 * (identity + r[0] * sigma_x + r[1] * sigma_y + r[2] * sigma_z)
                rho_eta = eta_f * rho + (1.0 - eta_f) * 0.5 * identity
                quantum_states_grouped.append([rho_eta])

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
    """Compute S_B^rac = average_{x,y} P(b=x_y|x,y)."""
    total = 0.0
    count = 0
    for x in range(scenario.X_cardinality):
        for y in range(scenario.Y_cardinality):
            total += float(scenario.data_numeric[x, y, 0, porac_bit(x, y)])
            count += 1
    return float(total / count)


def porac_keyrate_lower_bound(
    scenario: ContextualityScenario,
) -> tuple[float, float, float]:
    """Return (S_B^rac, S_E^rac, r_lb) with r_lb=-log2(S_E^rac)-h2(S_B^rac)."""
    s_b = porac_bob_success_probability(scenario)
    s_e = scenario.compute_eve_average_guessing_probability(
        guess_who="Bob",
        objective_label_map=porac_objective_label_map,
    )
    r_lb = float(-np.log2(s_e) - ContextualityScenario.binary_entropy(s_b))
    return float(s_b), float(s_e), float(r_lb)


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    # ---------------------------------------------------------------------
    # Build the noiseless PORAC scenario.
    # ---------------------------------------------------------------------
    scenario = build_porac_scenario(eta=1.0)
    ContextualityScenario.print_title("QKD: (3,2)-PORAC protocol (eta=1.000)")
    scenario.print_preparation_index_sets([(x,) for x in range(8)])

    print("\nSymbolic probability table p(b|x,y):")
    scenario.print_probabilities(as_p_b_given_x_y=True, precision=3, representation="symbolic")

    print("\nOperational equivalences:")
    scenario.print_operational_equivalences(precision=3, representation="symbolic")
    _print_porac_article_prep_opeqs()
    rank_article, rank_discovered, rank_stacked = _validate_porac_prep_opeq_subspace(scenario)
    print(
        "\nValidated PORAC prep-OPEQ subspace equivalence to article constraints: "
        f"rank(article)={rank_article}, rank(discovered)={rank_discovered}, rank(vstack)={rank_stacked}."
    )

    print("\nPORAC Eve-target guessing objective (e = x_y):")
    scenario.print_guessing_probability_grids(
        guess_who="Bob",
        precision=3,
        include_keyrate_pairs=False,
        objective_label_map=porac_objective_label_map,
    )

    # Alternative workaround (not used here):
    # One could build an auxiliary lifted table P_lift(a,b|x,y)=delta_{a,x_y} P(b|x,y)
    # and then call familiar guess_who="Alice" routines. We prefer the direct objective
    # map extension because it preserves the native PORAC semantics without an artificial A-lift.
    s_b, s_e, r_lb = porac_keyrate_lower_bound(scenario)
    print("\nPORAC key-rate components (Eq. 22):")
    print(f"S_B^rac = {ContextualityScenario.format_numeric(s_b, precision=6)}")
    print(f"S_E^rac = {ContextualityScenario.format_numeric(s_e, precision=6)}")
    print(
        "r_lb = -log2(S_E^rac) - h2(S_B^rac) = "
        f"{ContextualityScenario.format_numeric(r_lb, precision=6)}"
    )

    print("\nDepolarizing-noise sweep:")
    print("eta    S_B^rac   S_E^rac   r_lb")
    threshold_eta = None
    for eta in np.linspace(1.0, 0.6, 9):
        scenario_eta = build_porac_scenario(eta=float(eta))
        s_b_eta, s_e_eta, r_lb_eta = porac_keyrate_lower_bound(scenario_eta)
        if threshold_eta is None and r_lb_eta > 0:
            threshold_eta = float(eta)
        print(
            f"{ContextualityScenario.format_numeric(eta, precision=3):>5}  "
            f"{ContextualityScenario.format_numeric(s_b_eta, precision=6):>8}  "
            f"{ContextualityScenario.format_numeric(s_e_eta, precision=6):>8}  "
            f"{ContextualityScenario.format_numeric(r_lb_eta, precision=6):>8}"
        )
    if threshold_eta is None:
        print("No positive r_lb observed on this eta grid.")
    else:
        print(
            "First eta on grid with positive r_lb: "
            f"{ContextualityScenario.format_numeric(threshold_eta, precision=3)}"
        )


if __name__ == "__main__":
    main()

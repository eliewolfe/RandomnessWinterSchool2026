"""QKD protocol demo: (3,2)-PORAC with Bob-outcome protocol analysis."""

from __future__ import annotations

from pathlib import Path

import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import sympy as sp

from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.quantum import QuantumContextualityScenario
from contextualityqkd.scenario import ContextualityScenario


def _porac_index(x0: int, x1: int, x2: int) -> int:
    """Encode bit triple (x0,x1,x2) as integer x in {0,...,7}."""
    return int(4 * x0 + 2 * x1 + x2)


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
    rows = _porac_article_prep_opeq_rows()
    print(f"\nPreparation OPEQs according to the article ({rows.rows} rows):")
    for k in range(rows.rows):
        row_entries = [ContextualityScenario._format_symbolic_entry(rows[k, x], precision=6) for x in range(rows.cols)]
        ragged = "[[" + ", ".join(row_entries) + "]]"
        print(f"k={k}: {ragged}")


def _validate_porac_prep_opeq_subspace(scenario: ContextualityScenario) -> tuple[int, int, int]:
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
    """Construct Bob-outcome (3,2)-PORAC with 8 preparations and 3 binary measurements."""
    eta_f = float(eta)
    if eta_f < 0.0 or eta_f > 1.0:
        raise ValueError("eta must lie in [0,1].")

    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    paulis = [sigma_x, sigma_y, sigma_z]

    quantum_states: list[np.ndarray] = []
    for x0 in (0, 1):
        for x1 in (0, 1):
            for x2 in (0, 1):
                r = np.array([(-1) ** x0, (-1) ** x1, (-1) ** x2], dtype=float) / np.sqrt(3.0)
                rho = 0.5 * (identity + r[0] * sigma_x + r[1] * sigma_y + r[2] * sigma_z)
                rho_eta = eta_f * rho + (1.0 - eta_f) * 0.5 * identity
                quantum_states.append(rho_eta)

    quantum_effects_grouped: list[list[np.ndarray]] = []
    for y in range(3):
        plus = 0.5 * (identity + paulis[y])
        minus = 0.5 * (identity - paulis[y])
        quantum_effects_grouped.append([plus, minus])

    return QuantumContextualityScenario.from_quantum_states_effects(
        quantum_states=np.asarray(quantum_states, dtype=complex),
        quantum_effects=np.asarray(quantum_effects_grouped, dtype=complex),
        verbose=False,
    )


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    scenario = build_porac_scenario(eta=1.0)
    protocol = ContextualityProtocol(scenario, where_key=None)

    ContextualityScenario.print_title("QKD Protocol: (3,2)-PORAC (ideal noiseless case)")

    scenario.print_probabilities(as_p_b_given_x_y=True, precision=3, representation="symbolic")

    print("\nOperational equivalences:")
    scenario.print_operational_equivalences(precision=3, representation="symbolic")
    _print_porac_article_prep_opeqs()
    rank_article, rank_discovered, rank_stacked = _validate_porac_prep_opeq_subspace(scenario)
    print(
        "\nValidated PORAC prep-OPEQ subspace equivalence to article constraints: "
        f"rank(article)={rank_article}, rank(discovered)={rank_discovered}, rank(vstack)={rank_stacked}."
    )
    protocol.print_alice_guessing_metrics()
    protocol.print_alice_uncertainty_metrics()
    protocol.print_eve_guessing_metrics_lp()
    protocol.print_eve_uncertainty_metrics_reverse_fano_lp()
    protocol.print_key_rate_summary_reverse_fano_lp()

    auto_protocol = ContextualityProtocol(
        scenario,
        where_key="Automatic",
        optimize_verbose=True,
    )
    auto_protocol.print_where_key_optimization_best_stage(leading_newline=True)

    scenario.print_contextuality_measures(precision=3)


if __name__ == "__main__":
    main()

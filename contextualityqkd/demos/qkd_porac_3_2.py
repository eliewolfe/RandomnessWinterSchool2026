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


def _print_opeq_rows(title: str, rows: sp.Matrix, *, precision: int = 6) -> None:
    """Print OPEQ rows one per line, in the article's explicit coefficient form."""
    print(f"\n{title} ({rows.rows} rows):")
    for k in range(rows.rows):
        row_entries = [ContextualityScenario._format_symbolic_entry(rows[k, x], precision=precision) for x in range(rows.cols)]
        ragged = "[[" + ", ".join(row_entries) + "]]"
        print(f"k={k}: {ragged}")


def _print_porac_article_prep_opeqs() -> None:
    _print_opeq_rows("Preparation OPEQs according to the article", _porac_article_prep_opeq_rows())


def _float_rref(matrix: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Reduced row echelon form with partial pivoting, dropping numerically-zero rows.

    Done in floating point on purpose: the auto-discovered OPEQs arrive as an
    orthonormalised numerical basis, so exact elimination would treat ~1e-16
    round-off as genuine pivots.
    """
    work = np.array(matrix, dtype=float)
    n_rows, n_cols = work.shape
    pivot_row = 0
    for col in range(n_cols):
        if pivot_row >= n_rows:
            break
        candidate = pivot_row + int(np.argmax(np.abs(work[pivot_row:, col])))
        if abs(work[candidate, col]) <= tol:
            continue
        work[[pivot_row, candidate]] = work[[candidate, pivot_row]]
        work[pivot_row] /= work[pivot_row, col]
        for row in range(n_rows):
            if row != pivot_row:
                work[row] -= work[row, col] * work[pivot_row]
        pivot_row += 1
    return work[:pivot_row]


def _rationalize(matrix: np.ndarray, max_denominator: int = 64, tol: float = 1e-7) -> sp.Matrix:
    """Snap a float matrix back to exact rationals, rejecting entries that do not fit cleanly."""
    rows: list[list[sp.Rational]] = []
    for row in np.asarray(matrix, dtype=float):
        exact: list[sp.Rational] = []
        for value in row:
            candidate = sp.Rational(float(value)).limit_denominator(max_denominator)
            if abs(float(candidate) - float(value)) > tol:
                raise ValueError(
                    f"Entry {float(value)!r} is not a rational with denominator <= {max_denominator}."
                )
            exact.append(candidate)
        rows.append(exact)
    return sp.Matrix(rows)


def _canonical_opeq_rows(matrix: sp.Matrix) -> sp.Matrix:
    """Canonical exact form (rationalised RREF) of a set of OPEQ rows.

    Two OPEQ sets describe the same constraints exactly when their canonical
    forms are equal, whatever basis each was expressed in.
    """
    as_float = np.array(sp.Matrix(matrix).tolist(), dtype=float)
    return _rationalize(_float_rref(as_float))


def _span_residual(row: np.ndarray, basis: np.ndarray) -> float:
    """Largest coefficient error when least-squares fitting `row` inside `basis`'s row space."""
    design = np.asarray(basis, dtype=float).T
    target = np.asarray(row, dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(np.max(np.abs(design @ coefficients - target)))


def _validate_porac_prep_opeq_subspace(scenario: ContextualityScenario) -> tuple[sp.Matrix, sp.Matrix]:
    """Check the auto-discovered prep OPEQs against the article's, via exact canonical forms.

    Returns the (article, discovered) canonical forms so the caller can print them.
    """
    article = _porac_article_prep_opeq_rows()
    discovered = sp.Matrix(np.asarray(scenario.opeq_preps_symbolic, dtype=object).reshape(-1, scenario.X_cardinality))
    article_canonical = _canonical_opeq_rows(article)
    discovered_canonical = _canonical_opeq_rows(discovered)
    if article_canonical != discovered_canonical:
        raise ValueError(
            "Auto-discovered preparation OPEQs are not equivalent to PORAC article constraints:\n"
            f"article canonical form:\n{article_canonical}\n"
            f"discovered canonical form:\n{discovered_canonical}"
        )
    return article_canonical, discovered_canonical


def _print_porac_prep_opeq_comparison(scenario: ContextualityScenario) -> None:
    """Show the article OPEQs, the discovered ones in the same explicit form, and per-row membership."""
    article = _porac_article_prep_opeq_rows()
    discovered = sp.Matrix(np.asarray(scenario.opeq_preps_symbolic, dtype=object).reshape(-1, scenario.X_cardinality))

    _print_porac_article_prep_opeqs()

    article_canonical, discovered_canonical = _validate_porac_prep_opeq_subspace(scenario)
    _print_opeq_rows("Auto-discovered preparation OPEQs, canonical exact form", discovered_canonical)
    _print_opeq_rows("Article preparation OPEQs, canonical exact form", article_canonical)
    print(f"\nCanonical forms agree entry by entry: {article_canonical == discovered_canonical}")

    discovered_float = np.array(discovered.tolist(), dtype=float)
    print("\nEach article OPEQ, checked for membership in the auto-discovered span:")
    for k in range(article.rows):
        residual = _span_residual(np.array(article.row(k).tolist(), dtype=float).ravel(), discovered_float)
        verdict = "in span" if residual < 1e-9 else "NOT in span"
        print(f"k={k}: residual={residual:.2e}  ->  {verdict}")


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
    # Important caveat for comparing against the article:
    # - We run the nonprojective Naimark-unitary pathway (Bob/Eve not assumed
    #   projective in the solver constraints).
    # - The U-only generator trick is enabled to reduce SDP size while keeping
    #   the nonprojective constraint model active.
    protocol = ContextualityProtocol(
        scenario=scenario,
        # Optimal key selection (LP, reverse-Fano, objective = bits per key-generating run; exhaustive): no choice of
        # preparations per setting gives a positive LP rate (the best per-setting rate is exactly 0).
        # The positive rate reported below comes from the SDP bound only.
        where_key=None,
        master_key_holder="Alice",
        atol=1e-9,
        lp_solver="highs",
        sdp_solver="MOSEK",
        sdp_projective_bob=False,
        sdp_projective_eve=False,
        sdp_npa_level_bob=1,
        sdp_npa_level_eve=1,
        sdp_use_u_only=True,   # U-only generator set: smaller SDP, much faster
        sdp_threads=1,
        sdp_verbose=2,
    )
    # bob_protocol = ContextualityProtocol(
    #     scenario=scenario,
    #     where_key=None,
    #     master_key_holder="Bob",
    #     atol=1e-9,
    #     optimize_cluster_tolerance=1e-6,
    #     optimize_cluster_by="threshold_uncertainty",
    #     optimize_tie_break="earliest_optimal_stage",
    #     sdp_npa_level_bob=1,
    #     sdp_npa_level_eve=1,
    #     sdp_threads=1,
    #     sdp_verbose=2,
    # )

    ContextualityScenario.print_title("QKD Protocol: (3,2)-PORAC (ideal noiseless case)")

    scenario.print_probabilities(as_p_b_given_x_y=True, precision=3, representation="symbolic")

    print("\nOperational equivalences:")
    scenario.print_operational_equivalences(precision=3, representation="symbolic")
    scenario.print_contextuality_measures(metrics=["contextual_fraction"], precision=3, show_inequalities=True, backend_solver="mosek_simplex")
    _print_porac_prep_opeq_comparison(scenario)
    protocol.print_alice_guessing_metrics()
    protocol.print_alice_uncertainty_metrics()
    # bob_protocol.print_eve_security_metrics(
    #     method="both",
    #     rate_type="reverse_fano",
    #     include_per_y_lp=False,
    #     precision_vector=3,
    #     precision_scalar=6,
    #     leading_newline=True,
    # )
    protocol.print_eve_security_metrics(
        method="both",
        rate_type="reverse_fano",
        include_per_y_lp=False,
        precision_vector=3,
        precision_scalar=6,
        leading_newline=True,
    )
    protocol.print_eve_guess_upper_bound_inequality_by_y()
    protocol.print_eve_guess_upper_bound_inequality()

    # auto_protocol = ContextualityProtocol(
    #     scenario=scenario,
    #     where_key="Automatic",
    #     master_key_holder="Alice",
    #     atol=1e-9,
    #     optimize_cluster_tolerance=1e-6,
    #     optimize_cluster_by="threshold_uncertainty",
    #     optimize_tie_break="earliest_optimal_stage",
    #     sdp_npa_level_bob=1,
    #     sdp_npa_level_eve=1,
    #     sdp_threads=None,
    #     sdp_verbose=0,
    # )
    # auto_protocol.print_where_key_optimization_best_stage(leading_newline=True)


if __name__ == "__main__":
    main()

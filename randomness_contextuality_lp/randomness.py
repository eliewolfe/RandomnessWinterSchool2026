"""Native MOSEK LP routines for randomness quantification."""

from __future__ import annotations

import math

import mosek
import numpy as np

from .scenario import ContextualityScenario

def reverse_fano_bound(p_guess: float) -> float:
    """Return a lower bound on conditional Shannon entropy in bits from guessing probability."""
    p = float(p_guess)
    if p <= 0.0:
        raise ValueError("p_guess must be strictly positive.")
    if p < 1.0:
        p_eff = p
    else:
        p_eff = 1.0

    f = math.floor(1 / p_eff)
    c = f + 1
    return (c * p_eff - 1) * f * math.log2(f) + (1 - f * p_eff) * c * math.log2(c)

def min_entropy_bits(p_guess: float) -> float:
    """Return min-entropy in bits from guessing probability."""
    return float(-math.log2(p_guess))


def eve_optimal_guessing_probability(
    scenario: ContextualityScenario,
    x: int = 0,
    y: int = 0,
    bin_outcomes: list[list[int]] | tuple[tuple[int, ...], ...] | None = None,
) -> float:
    """Compute Eve's best guessing probability for one chosen target ``(x, y)``.

    Motivation
    ----------
    This quantifies certified unpredictability for a specific preparation/measurement
    choice. It is the targeted randomness metric to use when a protocol fixes one
    setting pair.

    How to use it with other functions
    ----------------------------------
    Build a ``ContextualityScenario`` first (directly, or via
    ``contextuality_scenario_from_gpt`` / ``contextuality_scenario_from_quantum``),
    then call this function with the desired ``x`` and ``y``. Convert the result to
    min-entropy with ``min_entropy_bits`` if needed.

    Input/output structure
    ----------------------
    Input is a validated ``ContextualityScenario`` and integer indices ``x`` and
    ``y`` in range. Set ``bin_outcomes`` to a partition of Bob outcomes (for
    example ``[[0, 1], [2, 3]]``) to guess bins rather than exact outcomes.
    Output is a single float in ``[0, 1]``: the LP optimum for
    Eve's guessing probability at that target pair.

    High-level implementation
    -------------------------
    Delegates to the shared LP builder/solver with a one-element target list,
    enforcing data consistency and operational-equivalence constraints from the
    scenario while maximizing Eve's success objective.
    """
    if x < 0 or x >= scenario.X_cardinality:
        raise ValueError(f"x must be in 0..{scenario.X_cardinality - 1}.")
    if y < 0 or y >= scenario.Y_cardinality:
        raise ValueError(f"y must be in 0..{scenario.Y_cardinality - 1}.")

    return _solve_guessing_lp(
        scenario=scenario,
        target_pairs=[(x, y)],
        bin_outcomes=bin_outcomes,
    )


def eve_optimal_average_guessing_probability(
    scenario: ContextualityScenario,
    bin_outcomes: list[list[int]] | tuple[tuple[int, ...], ...] | None = None,
) -> float:
    """Compute Eve's optimal guessing probability averaged over all ``(x, y)``.

    Motivation
    ----------
    This provides a global, setting-agnostic randomness figure when no single target
    pair is privileged.

    How to use it with other functions
    ----------------------------------
    Use the same scenario objects produced by ``ContextualityScenario`` constructors.
    Choose this function when you want one aggregate number; choose
    ``eve_optimal_guessing_probability`` when targeting a specific pair.

    Input/output structure
    ----------------------
    Input is one ``ContextualityScenario``. Set ``bin_outcomes`` to a partition
    of Bob outcomes (for example ``[[0, 1], [2, 3]]``) to optimize bin guessing
    rather than exact outcome guessing. Output is one float in ``[0, 1]``
    representing the LP optimum of the mean guessing objective over all settings.

    High-level implementation
    -------------------------
    Enumerates every ``(x, y)`` pair and solves one LP with an objective that
    averages per-target success terms, reusing the same feasibility constraints as
    the single-target optimization.
    """
    target_pairs = list(np.ndindex(scenario.X_cardinality, scenario.Y_cardinality))
    return _solve_guessing_lp(
        scenario=scenario,
        target_pairs=target_pairs,
        bin_outcomes=bin_outcomes,
    )


def analyze_scenario(
    scenario: ContextualityScenario,
    bin_outcomes: list[list[int]] | tuple[tuple[int, ...], ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute and cache Eve guessing and key-rate tables on ``scenario``.

    Returns
    -------
    tuple
        ``(p_guess_eve_table, keyrate_table)``, both with shape ``(X, Y)``.
    """
    num_x = scenario.X_cardinality
    num_y = scenario.Y_cardinality

    p_guess_eve_table = np.zeros((num_x, num_y), dtype=float)
    for x, y in np.ndindex(num_x, num_y):
        p_guess_eve = eve_optimal_guessing_probability(
            scenario,
            x=x,
            y=y,
            bin_outcomes=bin_outcomes,
        )
        p_guess_eve_table[x, y] = p_guess_eve

    keyrate_table = np.zeros_like(p_guess_eve_table)
    for x, y in np.ndindex(num_x, num_y):
        keyrate_table[x, y] = reverse_fano_bound(p_guess_eve_table[x, y]) - scenario.alice_conditional_entropy_table[x, y]

    scenario.p_guess_eve_table = p_guess_eve_table
    scenario.keyrate_table = keyrate_table
    return p_guess_eve_table, keyrate_table


def _solve_guessing_lp(
    scenario: ContextualityScenario,
    target_pairs: list[tuple[int, int]],
    bin_outcomes: list[list[int]] | tuple[tuple[int, ...], ...] | None = None,
) -> float:
    """Build and solve the LP for a list of target pairs."""
    data = scenario.data_numeric
    opeq_preps = scenario.opeq_preps_numeric
    opeq_meas = scenario.opeq_meas_numeric
    num_x = scenario.X_cardinality
    num_y = scenario.Y_cardinality
    num_a = scenario.A_cardinality
    num_b = scenario.B_cardinality
    bins = scenario._normalize_bob_outcome_bins(
        bin_outcomes=bin_outcomes,
        num_b=num_b,
    )
    num_e = len(bins)
    outcome_to_bin = np.empty(num_b, dtype=int)
    for bin_id, outcome_indices in enumerate(bins):
        outcome_to_bin[outcome_indices] = bin_id
    num_targets = len(target_pairs)

    num_variables = num_targets * num_x * num_y * num_a * num_b * num_e

    def var_index(
        t: int | np.ndarray,
        x: int | np.ndarray,
        y: int | np.ndarray,
        a: int | np.ndarray,
        b: int | np.ndarray,
        e: int | np.ndarray,
    ) -> int | np.ndarray:
        """Map multi-indices (with broadcasting) to flat MOSEK variable indices."""
        t_arr, x_arr, y_arr, a_arr, b_arr, e_arr = np.broadcast_arrays(t, x, y, a, b, e)
        idx = np.ravel_multi_index(
            (t_arr, x_arr, y_arr, a_arr, b_arr, e_arr),
            dims=(num_targets, num_x, num_y, num_a, num_b, num_e),
        )
        if np.ndim(idx) == 0:
            return int(idx)
        return idx.astype(int, copy=False)

    rows_cols: list[list[int]] = []
    rows_vals: list[list[float]] = []
    rhs: list[float] = []
    row_axis = slice(None), None
    coeff_axis = None, slice(None)

    # Data consistency:
    # sum_e P_t(a,b,e|x,y) = P_data(a,b|x,y)
    e_values = np.arange(num_e, dtype=int)
    e_ones = np.ones(num_e, dtype=float)
    for t, x, y, a, b in np.ndindex(num_targets, num_x, num_y, num_a, num_b):
        cols = var_index(t, x, y, a, b, e_values)
        rows_cols.append(cols.tolist())
        rows_vals.append(e_ones.tolist())
        rhs.append(float(data[x, y, a, b]))

    # Preparation operational equivalences:
    # sum_{x,a} c[x,a] P_t(a,b,e|x,y) = 0
    prep_row_shape = (num_targets, num_y, num_b, num_e)
    prep_rows_per_opeq = int(np.prod(prep_row_shape))
    prep_t, prep_y, prep_b, prep_e = np.unravel_index(
        np.arange(prep_rows_per_opeq, dtype=int),
        prep_row_shape,
    )
    for coeffs in opeq_preps:
        x_nonzero, a_nonzero = np.nonzero(coeffs)
        coeff_nonzero = coeffs[x_nonzero, a_nonzero].astype(float)

        if coeff_nonzero.size == 0:
            continue

        cols_matrix = var_index(
            prep_t[row_axis],
            x_nonzero[coeff_axis],
            prep_y[row_axis],
            a_nonzero[coeff_axis],
            prep_b[row_axis],
            prep_e[row_axis],
        )
        vals_matrix = np.broadcast_to(
            coeff_nonzero[coeff_axis],
            cols_matrix.shape,
        )
        rows_cols.extend(cols_matrix.tolist())
        rows_vals.extend(vals_matrix.tolist())
        rhs.extend([0.0] * prep_rows_per_opeq)

    # Measurement operational equivalences:
    # sum_{y,b} d[y,b] P_t(a,b,e|x,y) = 0
    meas_row_shape = (num_targets, num_x, num_a, num_e)
    meas_rows_per_opeq = int(np.prod(meas_row_shape))
    meas_t, meas_x, meas_a, meas_e = np.unravel_index(
        np.arange(meas_rows_per_opeq, dtype=int),
        meas_row_shape,
    )
    for coeffs in opeq_meas:
        y_nonzero, b_nonzero = np.nonzero(coeffs)
        coeff_nonzero = coeffs[y_nonzero, b_nonzero].astype(float)

        if coeff_nonzero.size == 0:
            continue

        cols_matrix = var_index(
            meas_t[row_axis],
            meas_x[row_axis],
            y_nonzero[coeff_axis],
            meas_a[row_axis],
            b_nonzero[coeff_axis],
            meas_e[row_axis],
        )
        vals_matrix = np.broadcast_to(
            coeff_nonzero[coeff_axis],
            cols_matrix.shape,
        )
        rows_cols.extend(cols_matrix.tolist())
        rows_vals.extend(vals_matrix.tolist())
        rhs.extend([0.0] * meas_rows_per_opeq)

    # Objective:
    # average over targets of sum_b sum_a P_t(a,b,e=bin(b)|x_t,y_t).
    obj = np.zeros(num_variables, dtype=float)
    weight = 1.0 / float(num_targets)
    for t, (target_x, target_y) in enumerate(target_pairs):
        for b, a in np.ndindex(num_b, num_a):
            guessed_e = int(outcome_to_bin[b])
            obj[var_index(t, target_x, target_y, a, b, guessed_e)] += weight

    with mosek.Env() as env:
        with env.Task(0, 0) as task:
            task.appendvars(num_variables)
            task.putvarboundsliceconst(0, num_variables, mosek.boundkey.lo, 0.0, 0.0)

            num_constraints = len(rows_cols)
            task.appendcons(num_constraints)
            for row_index in range(num_constraints):
                task.putarow(row_index, rows_cols[row_index], rows_vals[row_index])
                row_rhs = rhs[row_index]
                task.putconbound(row_index, mosek.boundkey.fx, row_rhs, row_rhs)

            obj_idx = [i for i, c in enumerate(obj) if c != 0.0]
            obj_val = [float(obj[i]) for i in obj_idx]
            if obj_idx:
                task.putclist(obj_idx, obj_val)

            task.putobjsense(mosek.objsense.maximize)
            task.optimize()

            acceptable_statuses = {mosek.solsta.optimal}
            if hasattr(mosek.solsta, "integer_optimal"):
                acceptable_statuses.add(mosek.solsta.integer_optimal)
            for soltype in (mosek.soltype.itr, mosek.soltype.bas):
                try:
                    solsta = task.getsolsta(soltype)
                except mosek.Error:
                    continue
                if solsta in acceptable_statuses:
                    return float(task.getprimalobj(soltype))

    raise RuntimeError("LP solve failed: MOSEK did not return an optimal solution.")

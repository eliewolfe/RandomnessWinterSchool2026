"""Native MOSEK LP routines for randomness quantification."""

from __future__ import annotations

import math

import mosek
import numpy as np

from .scenario import ContextualityScenario


def min_entropy_bits(p_guess: float) -> float:
    """Return min-entropy in bits from guessing probability."""
    return float(-math.log2(p_guess))


def eve_optimal_guessing_probability(
    scenario: ContextualityScenario,
    x: int = 0,
    y: int = 0,
) -> float:
    """Compute Eve's optimal guessing probability for one target pair ``(x, y)``."""
    if x < 0 or x >= scenario.X_cardinality:
        raise ValueError(f"x must be in 0..{scenario.X_cardinality - 1}.")
    if y < 0 or y >= scenario.Y_cardinality:
        raise ValueError(f"y must be in 0..{scenario.Y_cardinality - 1}.")

    return _solve_guessing_lp(scenario=scenario, target_pairs=[(x, y)])


def eve_optimal_average_guessing_probability(scenario: ContextualityScenario) -> float:
    """Compute average optimal guessing probability over all target pairs ``(x, y)``."""
    target_pairs = list(np.ndindex(scenario.X_cardinality, scenario.Y_cardinality))
    return _solve_guessing_lp(scenario=scenario, target_pairs=target_pairs)


def _solve_guessing_lp(
    scenario: ContextualityScenario,
    target_pairs: list[tuple[int, int]],
) -> float:
    """Build and solve the LP for a list of target pairs."""
    num_x = scenario.X_cardinality
    num_y = scenario.Y_cardinality
    num_a = scenario.A_cardinality
    num_b = scenario.B_cardinality
    num_e = num_b
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
        rhs.append(float(scenario.data[x, y, a, b]))

    # Preparation operational equivalences:
    # sum_{x,a} c[x,a] P_t(a,b,e|x,y) = 0
    prep_row_shape = (num_targets, num_y, num_b, num_e)
    prep_rows_per_opeq = int(np.prod(prep_row_shape))
    prep_t, prep_y, prep_b, prep_e = np.unravel_index(
        np.arange(prep_rows_per_opeq, dtype=int),
        prep_row_shape,
    )
    for coeffs in scenario.opeq_preps:
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
    for coeffs in scenario.opeq_meas:
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
    # average over targets of sum_b sum_a P_t(a,b,e=b|x_t,y_t).
    obj = np.zeros(num_variables, dtype=float)
    weight = 1.0 / float(num_targets)
    for t, (target_x, target_y) in enumerate(target_pairs):
        for b, a in np.ndindex(num_b, num_a):
            obj[var_index(t, target_x, target_y, a, b, b)] += weight

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

            for soltype in (mosek.soltype.itr, mosek.soltype.bas):
                try:
                    solsta = task.getsolsta(soltype)
                except mosek.Error:
                    continue
                if solsta in (mosek.solsta.optimal, mosek.solsta.near_optimal):
                    return float(task.getprimalobj(soltype))

    raise RuntimeError("LP solve failed: MOSEK did not return an optimal solution.")

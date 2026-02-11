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
        idx = (
            (((((np.asarray(t) * num_x) + np.asarray(x)) * num_y + np.asarray(y)) * num_a + np.asarray(a))
             * num_b + np.asarray(b))
            * num_e
            + np.asarray(e)
        )
        idx = np.asarray(idx, dtype=int)
        if idx.ndim == 0:
            return int(idx)
        return idx

    rows_cols: list[list[int]] = []
    rows_vals: list[list[float]] = []
    rhs: list[float] = []

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
    for t, k, y, b, e in np.ndindex(
        num_targets,
        scenario.opeq_preps.shape[0],
        num_y,
        num_b,
        num_e,
    ):
        coeffs = scenario.opeq_preps[k]
        x_nonzero, a_nonzero = np.nonzero(coeffs)
        coeff_nonzero = coeffs[x_nonzero, a_nonzero].astype(float)
        cols = var_index(t, x_nonzero, y, a_nonzero, b, e)
        rows_cols.append(cols.tolist())
        rows_vals.append(coeff_nonzero.tolist())
        rhs.append(0.0)

    # Measurement operational equivalences:
    # sum_{y,b} d[y,b] P_t(a,b,e|x,y) = 0
    for t, k, x, a, e in np.ndindex(
        num_targets,
        scenario.opeq_meas.shape[0],
        num_x,
        num_a,
        num_e,
    ):
        coeffs = scenario.opeq_meas[k]
        y_nonzero, b_nonzero = np.nonzero(coeffs)
        coeff_nonzero = coeffs[y_nonzero, b_nonzero].astype(float)
        cols = var_index(t, x, y_nonzero, a, b_nonzero, e)
        rows_cols.append(cols.tolist())
        rows_vals.append(coeff_nonzero.tolist())
        rhs.append(0.0)

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

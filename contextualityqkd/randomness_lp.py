"""Bob-outcome LP backends."""

from __future__ import annotations

from typing import Sequence

import mosek
import numpy as np

from .scenario import ContextualityScenario

def _build_bob_single_model_lp_components(
    scenario: ContextualityScenario,
):
    """Build shared single-model LP components for Bob-outcome Eve guessing."""
    data = scenario.data_numeric
    opeq_preps = scenario.opeq_preps_numeric
    opeq_meas = scenario.opeq_meas_numeric
    num_x = scenario.X_cardinality
    num_y = scenario.Y_cardinality
    num_b = scenario.B_cardinality
    b_cardinality_per_y = scenario.b_cardinality_per_y.astype(int, copy=False)
    num_e = num_b

    num_variables = num_x * num_y * num_b * num_e

    def var_index(
        x: int | np.ndarray,
        y: int | np.ndarray,
        b: int | np.ndarray,
        e: int | np.ndarray,
    ) -> int | np.ndarray:
        x_arr, y_arr, b_arr, e_arr = np.broadcast_arrays(x, y, b, e)
        idx = np.ravel_multi_index(
            (x_arr, y_arr, b_arr, e_arr),
            dims=(num_x, num_y, num_b, num_e),
        )
        if np.ndim(idx) == 0:
            return int(idx)
        return idx.astype(int, copy=False)

    rows_cols: list[list[int]] = []
    rows_vals: list[list[float]] = []
    rhs: list[float] = []

    def append_var_zero_constraint(index: int) -> None:
        rows_cols.append([int(index)])
        rows_vals.append([1.0])
        rhs.append(0.0)

    # Data consistency: sum_e P_t(b,e|x,y) = p_data(b|x,y)
    e_values = np.arange(num_e, dtype=int)
    e_ones = np.ones(num_e, dtype=float)
    for x, y, b in np.ndindex(num_x, num_y, num_b):
        cols = var_index(x, y, b, e_values)
        rows_cols.append(cols.tolist())
        rows_vals.append(e_ones.tolist())
        rhs.append(float(data[x, y, b]))

    # Preparation OPEQs: sum_x c[x] P_t(b,e|x,y) = 0
    row_axis = (slice(None), None)
    coeff_axis = (None, slice(None))
    prep_row_shape = (num_y, num_b, num_e)
    prep_rows_per_opeq = int(np.prod(prep_row_shape))
    prep_y, prep_b, prep_e = np.unravel_index(
        np.arange(prep_rows_per_opeq, dtype=int),
        prep_row_shape,
    )
    for coeffs in opeq_preps:
        x_nonzero = np.flatnonzero(np.asarray(coeffs, dtype=float))
        coeff_nonzero = np.asarray(coeffs, dtype=float)[x_nonzero]
        if coeff_nonzero.size == 0:
            continue

        cols_matrix = var_index(
            x_nonzero[coeff_axis],
            prep_y[row_axis],
            prep_b[row_axis],
            prep_e[row_axis],
        )
        vals_matrix = np.broadcast_to(coeff_nonzero[coeff_axis], cols_matrix.shape)
        rows_cols.extend(cols_matrix.tolist())
        rows_vals.extend(vals_matrix.tolist())
        rhs.extend([0.0] * prep_rows_per_opeq)

    # Measurement OPEQs: sum_{y,b} d[y,b] P_t(b,e|x,y) = 0
    meas_row_shape = (num_x, num_e)
    meas_rows_per_opeq = int(np.prod(meas_row_shape))
    meas_x, meas_e = np.unravel_index(np.arange(meas_rows_per_opeq, dtype=int), meas_row_shape)
    for coeffs in opeq_meas:
        y_nonzero, b_nonzero = np.nonzero(np.asarray(coeffs, dtype=float))
        coeff_nonzero = np.asarray(coeffs, dtype=float)[y_nonzero, b_nonzero]
        if coeff_nonzero.size == 0:
            continue

        cols_matrix = var_index(
            meas_x[row_axis],
            y_nonzero[coeff_axis],
            b_nonzero[coeff_axis],
            meas_e[row_axis],
        )
        vals_matrix = np.broadcast_to(coeff_nonzero[coeff_axis], cols_matrix.shape)
        rows_cols.extend(cols_matrix.tolist())
        rows_vals.extend(vals_matrix.tolist())
        rhs.extend([0.0] * meas_rows_per_opeq)

    # Enforce invalid guess labels to carry zero mass when B(y) < B_max.
    for x, y in np.ndindex(num_x, num_y):
        b_count = int(b_cardinality_per_y[y])
        if b_count >= num_e:
            continue
        invalid_e = np.arange(b_count, num_e, dtype=int)
        for b in range(b_count):
            cols = var_index(x, y, b, invalid_e)
            for idx in np.asarray(cols, dtype=int).tolist():
                append_var_zero_constraint(int(idx))

    return (
        num_x,
        num_y,
        num_b,
        b_cardinality_per_y,
        num_e,
        num_variables,
        var_index,
        rows_cols,
        rows_vals,
        rhs,
    )


def _solve_eve_guess_bob_by_y_lp_hotstart(
    scenario: ContextualityScenario,
    where_key: Sequence[Sequence[int]],
) -> np.ndarray:
    """Solve Eve LP for Bob guessing probability for each y under key-conditioning subsets."""
    (
        num_x,
        num_y,
        _num_b,
        b_cardinality_per_y,
        _num_e,
        num_variables,
        var_index,
        rows_cols,
        rows_vals,
        rhs,
    ) = _build_bob_single_model_lp_components(scenario=scenario)

    if len(where_key) != num_y:
        raise ValueError(f"where_key must have one row per y (expected {num_y}).")

    objective_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for y, row in enumerate(where_key):
        x_row = np.asarray(row, dtype=int).reshape(-1)
        if x_row.size == 0:
            objective_data[y] = (
                np.empty((0,), dtype=int),
                np.empty((0,), dtype=float),
            )
            continue
        if np.any(x_row < 0) or np.any(x_row >= num_x):
            raise ValueError(f"where_key[{y}] contains out-of-range x index.")

        weight = 1.0 / float(x_row.size)
        coeffs: dict[int, float] = {}
        b_count = int(b_cardinality_per_y[y])
        for x in x_row.tolist():
            for b in range(b_count):
                idx = int(var_index(x, y, b, b))
                coeffs[idx] = coeffs.get(idx, 0.0) + weight

        idx = np.asarray(list(coeffs.keys()), dtype=int)
        val = np.asarray([coeffs[int(i)] for i in idx.tolist()], dtype=float)
        objective_data[y] = (idx, val)

    out = np.full((num_y,), np.nan, dtype=float)
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

            task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.primal_simplex)
            task.putintparam(mosek.iparam.sim_hotstart, mosek.simhotstart.status_keys)
            task.putobjsense(mosek.objsense.maximize)

            previous_idx: np.ndarray | None = None
            for y in range(num_y):
                if previous_idx is not None and previous_idx.size:
                    task.putclist(previous_idx.tolist(), [0.0] * int(previous_idx.size))

                idx, val = objective_data[y]
                if idx.size == 0:
                    previous_idx = idx
                    continue

                task.putclist(idx.tolist(), val.tolist())
                termination_code = task.optimize()
                out[y] = _get_optimal_primal_objective(
                    task,
                    failure_context=(f"Bob-guessing LP for y={y}"),
                    termination_code=termination_code,
                )
                previous_idx = idx

    return out


def _get_optimal_primal_objective(
    task: mosek.Task,
    failure_context: str | None = None,
    termination_code: object | None = None,
) -> float:
    """Extract an optimal LP objective from MOSEK task solutions."""
    acceptable_statuses = {mosek.solsta.optimal}
    if hasattr(mosek.solsta, "integer_optimal"):
        acceptable_statuses.add(mosek.solsta.integer_optimal)
    status_report: list[str] = []
    for soltype in (mosek.soltype.itr, mosek.soltype.bas):
        try:
            solsta = task.getsolsta(soltype)
        except mosek.Error:
            status_report.append(f"{soltype}: solsta unavailable")
            continue
        try:
            prosta = task.getprosta(soltype)
        except mosek.Error:
            prosta = "unavailable"
        status_report.append(f"{soltype}: solsta={solsta}, prosta={prosta}")
        if solsta in acceptable_statuses:
            return float(task.getprimalobj(soltype))
    context_prefix = f"{failure_context}. " if failure_context else ""
    trm = f" termination={termination_code}." if termination_code is not None else ""
    status_text = " | ".join(status_report) if status_report else "no solution-status information"
    raise RuntimeError(
        f"{context_prefix}LP solve failed: MOSEK did not return an optimal solution.{trm} "
        f"statuses: {status_text}"
    )

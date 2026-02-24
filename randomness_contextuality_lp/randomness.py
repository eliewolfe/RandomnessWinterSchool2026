"""Native MOSEK LP routines for randomness quantification."""

from __future__ import annotations

import math
from typing import Literal

import mosek
import numpy as np

from .scenario import ContextualityScenario


GuessWho = Literal["Bob", "Alice", "Both"]


def _normalize_guess_who(guess_who: str | None) -> GuessWho:
    """Normalize guess target selector to canonical capitalization."""
    if guess_who is None:
        return "Bob"

    normalized = str(guess_who).strip().lower()
    if normalized == "bob":
        return "Bob"
    if normalized == "alice":
        return "Alice"
    if normalized == "both":
        return "Both"
    raise ValueError("guess_who must be one of 'Bob', 'Alice', or 'Both' (case-insensitive).")


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


def min_entropy(p_guess: float) -> float:
    """Return min-entropy in bits from guessing probability."""
    return float(-math.log2(p_guess))


def eve_optimal_guessing_probability(
    scenario: ContextualityScenario,
    x: int = 0,
    y: int = 0,
    guess_who: str = "Bob",
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
    min-entropy with ``min_entropy`` if needed.

    Input/output structure
    ----------------------
    Input is a validated ``ContextualityScenario`` and integer indices ``x`` and
    ``y`` in range. Set ``guess_who`` to ``"Bob"``, ``"Alice"``, or ``"Both"``
    (case-insensitive). Output is a single float in ``[0, 1]``: the LP optimum
    for Eve's guessing probability at that target pair.

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

    target = _normalize_guess_who(guess_who)
    p_guess = _solve_guessing_lp_aggregate_objective(
        scenario=scenario,
        objective_terms=[(x, y, 1.0)],
        guess_who=target,
    )
    return float(p_guess)


def eve_optimal_average_guessing_probability(
    scenario: ContextualityScenario,
    guess_who: str = "Bob",
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
    Input is one ``ContextualityScenario``. Set ``guess_who`` to ``"Bob"``,
    ``"Alice"``, or ``"Both"`` (case-insensitive). Output is one float in
    ``[0, 1]`` representing the LP optimum of the mean guessing objective over
    all settings.

    High-level implementation
    -------------------------
    Enumerates every ``(x, y)`` pair and solves one LP with an objective that
    averages per-target success terms, reusing the same feasibility constraints as
    the single-target optimization.
    """
    num_targets = float(scenario.X_cardinality * scenario.Y_cardinality)
    objective_terms = [
        (x, y, 1.0 / num_targets)
        for x, y in np.ndindex(scenario.X_cardinality, scenario.Y_cardinality)
    ]
    target = _normalize_guess_who(guess_who)
    p_guess = _solve_guessing_lp_aggregate_objective(
        scenario=scenario,
        objective_terms=objective_terms,
        guess_who=target,
    )
    return float(p_guess)


def analyze_scenario(
    scenario: ContextualityScenario,
    guess_who: str = "Bob",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute and cache Eve guessing and key-rate tables on ``scenario``.

    Returns
    -------
    tuple
        ``(p_guess_eve_table, keyrate_table)``, both with shape ``(X, Y)``.
    """
    target = _normalize_guess_who(guess_who)
    num_x = scenario.X_cardinality
    num_y = scenario.Y_cardinality

    p_guess_eve_table = _solve_guessing_lp_hotstart_table(
        scenario=scenario,
        guess_who=target,
    )

    if target == "Bob":
        conditional_entropy_table = scenario.conditional_entropy_table_bob_given_alice
    elif target == "Alice":
        conditional_entropy_table = scenario.conditional_entropy_table_alice_given_bob
    else:
        conditional_entropy_table = scenario.conditional_entropy_table_alice_and_bob

    keyrate_table = np.zeros_like(p_guess_eve_table)
    for x, y in np.ndindex(num_x, num_y):
        keyrate_table[x, y] = reverse_fano_bound(p_guess_eve_table[x, y]) - conditional_entropy_table[x, y]

    scenario.set_p_guess_eve_table(p_guess_eve_table, guess_who=target)
    scenario.set_keyrate_table(keyrate_table, guess_who=target)
    return p_guess_eve_table, keyrate_table


def _build_single_model_lp_components(
    scenario: ContextualityScenario,
    guess_who: GuessWho,
):
    """Build shared single-model LP components used by sweep/single/average objectives."""
    data = scenario.data_numeric
    opeq_preps = scenario.opeq_preps_numeric
    opeq_meas = scenario.opeq_meas_numeric
    num_x = scenario.X_cardinality
    num_y = scenario.Y_cardinality
    num_a = scenario.A_cardinality
    num_b = scenario.B_cardinality
    a_cardinality_per_x = scenario.a_cardinality_per_x.astype(int, copy=False)
    b_cardinality_per_y = scenario.b_cardinality_per_y.astype(int, copy=False)
    if guess_who == "Bob":
        num_e = num_b
    elif guess_who == "Alice":
        num_e = num_a
    elif guess_who == "Both":
        num_e = num_a * num_b
    else:
        raise ValueError("Unsupported guess_who value.")

    num_variables = num_x * num_y * num_a * num_b * num_e

    def guess_event_index(a: int, b: int) -> int:
        if guess_who == "Bob":
            return int(b)
        if guess_who == "Alice":
            return int(a)
        return int(a * num_b + b)

    def var_index(
        x: int | np.ndarray,
        y: int | np.ndarray,
        a: int | np.ndarray,
        b: int | np.ndarray,
        e: int | np.ndarray,
    ) -> int | np.ndarray:
        x_arr, y_arr, a_arr, b_arr, e_arr = np.broadcast_arrays(x, y, a, b, e)
        idx = np.ravel_multi_index(
            (x_arr, y_arr, a_arr, b_arr, e_arr),
            dims=(num_x, num_y, num_a, num_b, num_e),
        )
        if np.ndim(idx) == 0:
            return int(idx)
        return idx.astype(int, copy=False)

    rows_cols: list[list[int]] = []
    rows_vals: list[list[float]] = []
    rhs: list[float] = []
    row_axis = slice(None), None
    coeff_axis = None, slice(None)

    def append_var_zero_constraint(index: int) -> None:
        rows_cols.append([int(index)])
        rows_vals.append([1.0])
        rhs.append(0.0)

    # Data consistency: sum_e P(a,b,e|x,y) = P_data(a,b|x,y)
    e_values = np.arange(num_e, dtype=int)
    e_ones = np.ones(num_e, dtype=float)
    for x, y, a, b in np.ndindex(num_x, num_y, num_a, num_b):
        cols = var_index(x, y, a, b, e_values)
        rows_cols.append(cols.tolist())
        rows_vals.append(e_ones.tolist())
        rhs.append(float(data[x, y, a, b]))

    # Preparation OPEQs: sum_{x,a} c[x,a] P(a,b,e|x,y) = 0
    prep_row_shape = (num_y, num_b, num_e)
    prep_rows_per_opeq = int(np.prod(prep_row_shape))
    prep_y, prep_b, prep_e = np.unravel_index(
        np.arange(prep_rows_per_opeq, dtype=int),
        prep_row_shape,
    )
    for coeffs in opeq_preps:
        x_nonzero, a_nonzero = np.nonzero(coeffs)
        coeff_nonzero = coeffs[x_nonzero, a_nonzero].astype(float)
        if coeff_nonzero.size == 0:
            continue

        cols_matrix = var_index(
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

    # Measurement OPEQs: sum_{y,b} d[y,b] P(a,b,e|x,y) = 0
    meas_row_shape = (num_x, num_a, num_e)
    meas_rows_per_opeq = int(np.prod(meas_row_shape))
    meas_x, meas_a, meas_e = np.unravel_index(
        np.arange(meas_rows_per_opeq, dtype=int),
        meas_row_shape,
    )
    for coeffs in opeq_meas:
        y_nonzero, b_nonzero = np.nonzero(coeffs)
        coeff_nonzero = coeffs[y_nonzero, b_nonzero].astype(float)
        if coeff_nonzero.size == 0:
            continue

        cols_matrix = var_index(
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

    # Enforce invalid guess labels to carry zero mass for each target setting pair.
    if guess_who == "Bob":
        for x, y in np.ndindex(num_x, num_y):
            b_count = int(b_cardinality_per_y[y])
            if b_count >= num_e:
                continue
            invalid_e = np.arange(b_count, num_e, dtype=int)
            for a in range(int(a_cardinality_per_x[x])):
                for b in range(b_count):
                    cols = var_index(x, y, a, b, invalid_e)
                    for idx in np.asarray(cols, dtype=int).tolist():
                        append_var_zero_constraint(int(idx))
    elif guess_who == "Alice":
        for x, y in np.ndindex(num_x, num_y):
            a_count = int(a_cardinality_per_x[x])
            if a_count >= num_e:
                continue
            invalid_e = np.arange(a_count, num_e, dtype=int)
            for a in range(a_count):
                for b in range(int(b_cardinality_per_y[y])):
                    cols = var_index(x, y, a, b, invalid_e)
                    for idx in np.asarray(cols, dtype=int).tolist():
                        append_var_zero_constraint(int(idx))
    elif guess_who == "Both":
        for x, y in np.ndindex(num_x, num_y):
            a_count = int(a_cardinality_per_x[x])
            b_count = int(b_cardinality_per_y[y])
            invalid_e: list[int] = []
            for e in range(num_e):
                guess_a = int(e // num_b)
                guess_b = int(e % num_b)
                if guess_a >= a_count or guess_b >= b_count:
                    invalid_e.append(e)
            if not invalid_e:
                continue
            invalid_e_arr = np.asarray(invalid_e, dtype=int)
            for a in range(a_count):
                for b in range(b_count):
                    cols = var_index(x, y, a, b, invalid_e_arr)
                    for idx in np.asarray(cols, dtype=int).tolist():
                        append_var_zero_constraint(int(idx))

    return (
        num_x,
        num_y,
        num_a,
        num_b,
        a_cardinality_per_x,
        b_cardinality_per_y,
        num_e,
        num_variables,
        var_index,
        rows_cols,
        rows_vals,
        rhs,
        guess_event_index,
    )


def _solve_guessing_lp_hotstart_table(
    scenario: ContextualityScenario,
    target_pairs: list[tuple[int, int]] | None = None,
    guess_who: GuessWho = "Bob",
) -> np.ndarray:
    """Solve selected single-target Eve objectives by re-optimizing one primal simplex task."""
    (
        num_x,
        num_y,
        num_a,
        num_b,
        a_cardinality_per_x,
        b_cardinality_per_y,
        _num_e,
        num_variables,
        var_index,
        rows_cols,
        rows_vals,
        rhs,
        guess_event_index,
    ) = _build_single_model_lp_components(
        scenario=scenario,
        guess_who=guess_who,
    )

    if target_pairs is None:
        target_pairs_list = list(np.ndindex(num_x, num_y))
    else:
        target_pairs_list = [(int(x), int(y)) for x, y in target_pairs]
        for target_x, target_y in target_pairs_list:
            if target_x < 0 or target_x >= num_x:
                raise ValueError(f"x must be in 0..{num_x - 1}.")
            if target_y < 0 or target_y >= num_y:
                raise ValueError(f"y must be in 0..{num_y - 1}.")

    objective_indices: dict[tuple[int, int], np.ndarray] = {}
    for target_x, target_y in target_pairs_list:
        idx_list: list[int] = []
        for a in range(int(a_cardinality_per_x[target_x])):
            for b in range(int(b_cardinality_per_y[target_y])):
                guessed_e = guess_event_index(a, b)
                idx_list.append(var_index(target_x, target_y, a, b, guessed_e))
        objective_indices[(target_x, target_y)] = np.asarray(idx_list, dtype=int)

    p_guess_table = np.full((num_x, num_y), np.nan, dtype=float)
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
            for target_x, target_y in target_pairs_list:
                if previous_idx is not None and previous_idx.size:
                    task.putclist(previous_idx.tolist(), [0.0] * int(previous_idx.size))

                current_idx = objective_indices[(target_x, target_y)]
                task.putclist(current_idx.tolist(), [1.0] * int(current_idx.size))
                termination_code = task.optimize()
                p_guess_table[target_x, target_y] = _get_optimal_primal_objective(
                    task,
                    failure_context=(
                        f"single-target LP for (x={target_x}, y={target_y})"
                    ),
                    termination_code=termination_code,
                )
                previous_idx = current_idx
    return p_guess_table


def _solve_guessing_lp_aggregate_objective(
    scenario: ContextualityScenario,
    objective_terms: list[tuple[int, int, float]],
    guess_who: GuessWho = "Bob",
) -> float:
    """Solve one LP with a weighted aggregate objective over target settings."""
    (
        num_x,
        num_y,
        num_a,
        num_b,
        a_cardinality_per_x,
        b_cardinality_per_y,
        _num_e,
        num_variables,
        var_index,
        rows_cols,
        rows_vals,
        rhs,
        guess_event_index,
    ) = _build_single_model_lp_components(
        scenario=scenario,
        guess_who=guess_who,
    )

    if len(objective_terms) == 0:
        raise ValueError("objective_terms must be non-empty.")

    objective_coeffs: dict[int, float] = {}
    for target_x, target_y, weight in objective_terms:
        x = int(target_x)
        y = int(target_y)
        w = float(weight)
        if x < 0 or x >= num_x:
            raise ValueError(f"x must be in 0..{num_x - 1}.")
        if y < 0 or y >= num_y:
            raise ValueError(f"y must be in 0..{num_y - 1}.")
        for a in range(int(a_cardinality_per_x[x])):
            for b in range(int(b_cardinality_per_y[y])):
                guessed_e = guess_event_index(a, b)
                idx = int(var_index(x, y, a, b, guessed_e))
                objective_coeffs[idx] = objective_coeffs.get(idx, 0.0) + w

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

            if objective_coeffs:
                obj_idx = list(objective_coeffs.keys())
                obj_val = [float(objective_coeffs[idx]) for idx in obj_idx]
                task.putclist(obj_idx, obj_val)

            task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.primal_simplex)
            task.putobjsense(mosek.objsense.maximize)
            termination_code = task.optimize()
            return _get_optimal_primal_objective(
                task,
                failure_context=(
                    f"aggregate-objective LP (num_terms={len(objective_terms)})"
                ),
                termination_code=termination_code,
            )


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

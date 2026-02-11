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
    ``y`` in range. Output is a single float in ``[0, 1]``: the LP optimum for
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

    return _solve_guessing_lp(scenario=scenario, target_pairs=[(x, y)])


def eve_optimal_average_guessing_probability(scenario: ContextualityScenario) -> float:
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
    Input is one ``ContextualityScenario``. Output is one float in ``[0, 1]``
    representing the LP optimum of the mean guessing objective over all settings.

    High-level implementation
    -------------------------
    Enumerates every ``(x, y)`` pair and solves one LP with an objective that
    averages per-target success terms, reusing the same feasibility constraints as
    the single-target optimization.
    """
    target_pairs = list(np.ndindex(scenario.X_cardinality, scenario.Y_cardinality))
    return _solve_guessing_lp(scenario=scenario, target_pairs=target_pairs)


def run_gpt_example(
    gpt_states: np.ndarray,
    gpt_effect_set: np.ndarray,
    title: str | None = None,
    target_pair: tuple[int, int] | None = None,
    source_outcome_distribution: np.ndarray | None = None,
    unit_effect: np.ndarray | None = None,
    atol: float = 1e-9,
    outcomes_per_measurement: int = 2,
    drop_tiny_imag: bool = True,
    verbose: bool = True,
) -> tuple[ContextualityScenario, list[tuple[int, ...]], float]:
    """Convenience pipeline from GPT vectors to scenario and randomness output.

    Motivation
    ----------
    This is the GPT-side quick-start helper for exploratory workflows. It builds the
    contextuality scenario, reports inferred measurement groupings from a flat effect
    set, and optionally evaluates a targeted Eve-guessing objective.

    How to use it with other functions
    ----------------------------------
    Internally this wraps ``contextuality_scenario_from_gpt`` and, when a target
    measurement is provided, calls ``eve_optimal_guessing_probability``. Use it for
    single-call experiments; use lower-level constructors/optimizers directly when
    you want finer control.

    Parameters
    ----------
    gpt_states:
        GPT preparation vectors with shape ``(X,K)`` or ``(X,A,K)``.
    gpt_effect_set:
        Flat GPT effect set with shape ``(N_effects,K)``.
    title:
        Optional heading printed before the report.
    target_pair:
        Optional target setting tuple ``(x,y)``. If provided, the function reports
        Eve's optimal guessing probability for that ``x`` and ``y``.
    source_outcome_distribution:
        Optional ``P(a|x)`` used to form joint ``P(a,b|x,y)`` when states are
        provided without explicit source outcomes.
    unit_effect:
        Optional GPT unit-effect vector; inferred from dimension when omitted.
    atol:
        Numerical tolerance used in inference/validation.
    outcomes_per_measurement:
        Number of outcomes per inferred measurement subset.
    drop_tiny_imag:
        If True, drop tiny imaginary components introduced by numerical noise.
    verbose:
        If True, scenario constructor prints probabilities and OPEQs.

    Returns
    -------
    tuple
        ``(scenario, measurement_indices, p_guess)``.

    High-level implementation
    -------------------------
    The function first builds a ``ContextualityScenario`` from GPT data, then solves
    the single-target Eve LP at ``(x,y)`` from ``target_pair`` or defaults to
    ``(0,0)``.
    """
    from .quantum import contextuality_scenario_from_gpt

    if title is not None:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    scenario, measurement_indices = contextuality_scenario_from_gpt(
        gpt_states=gpt_states,
        gpt_effect_set=gpt_effect_set,
        source_outcome_distribution=source_outcome_distribution,
        unit_effect=unit_effect,
        atol=atol,
        outcomes_per_measurement=outcomes_per_measurement,
        drop_tiny_imag=drop_tiny_imag,
        verbose=verbose,
        return_measurement_indices=True,
    )

    if target_pair is None:
        x_target, y_target = 0, 0
    else:
        x_target, y_target = target_pair

    p_guess = eve_optimal_guessing_probability(scenario, x=x_target, y=y_target)
    print("\nEve optimal guessing probability for target setting:")
    print(f"x_target={x_target}, y_target={y_target}")
    print(f"measurement indices={measurement_indices[y_target]}")
    print(f"P_guess = {p_guess:.10f}")
    return scenario, measurement_indices, p_guess


def run_quantum_example(
    quantum_states: np.ndarray,
    quantum_effect_set: np.ndarray,
    title: str | None = None,
    target_pair: tuple[int, int] | None = None,
    outcomes_per_measurement: int = 2,
    verbose: bool = True,
) -> tuple[ContextualityScenario, list[tuple[int, ...]], float]:
    """Convenience pipeline from quantum objects to scenario and randomness output.

    Motivation
    ----------
    This is a quick-start helper for experiments and demos: one call builds the
    contextuality scenario, exposes inferred measurement groupings, and optionally
    computes a targeted guessing probability.

    How to use it with other functions
    ----------------------------------
    Internally this converts quantum matrices into GPT vectors and delegates to
    ``run_gpt_example``. It is best for interactive exploration; for production
    flows you may call lower-level constructors/optimizers directly.

    Parameters
    ----------
    quantum_states:
        Quantum states with shape ``(X,d,d)`` or ``(X,A,d,d)``.
    quantum_effect_set:
        Flat set of effects with shape ``(N_effects,d,d)``.
    title:
        Optional heading printed before the report.
    target_pair:
        Optional target setting tuple ``(x,y)``. If provided, the function reports
        Eve's optimal guessing probability for that ``x`` and ``y``.
    outcomes_per_measurement:
        Number of outcomes per inferred measurement subset.
    verbose:
        If True, scenario constructor prints probabilities and OPEQs.

    Returns
    -------
    tuple
        ``(scenario, measurement_indices, p_guess)``.

    High-level implementation
    -------------------------
    The function converts matrices to GPT vectors in a Hilbert-Schmidt basis and
    then executes the same scenario-build and optional target-evaluation pipeline as
    ``run_gpt_example``. If ``target_pair`` is omitted, it defaults to ``(0,0)``.
    """
    from .quantum import convert_matrix_list_to_vector_list

    gpt_states = convert_matrix_list_to_vector_list(quantum_states)
    gpt_effect_set = convert_matrix_list_to_vector_list(quantum_effect_set)
    return run_gpt_example(
        gpt_states=gpt_states,
        gpt_effect_set=gpt_effect_set,
        title=title,
        target_pair=target_pair,
        outcomes_per_measurement=outcomes_per_measurement,
        verbose=verbose,
    )


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

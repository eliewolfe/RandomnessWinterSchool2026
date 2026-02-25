"""Simplex-embeddability and contextuality quantifiers for scenarios.

This module uses assignment-ray products to build LPs for:
1. Dephasing robustness (minimum dephasing to enter the noncontextual cone),
2. Contextual fraction (maximum noncontextual subbehavior mass).
"""

from __future__ import annotations

from dataclasses import dataclass

import mosek
import numpy as np
from scipy.sparse import csc_matrix

from .linalg_utils import enumerate_cone_extremal_rays, select_linearly_independent_rows
from .scenario import ContextualityScenario


@dataclass
class SimplexEmbeddabilityResult:
    """Result bundle for simplex-embeddability assessment."""

    is_simplex_embeddable: bool
    dephasing_robustness: float
    preparation_extremals: np.ndarray
    effect_extremals: np.ndarray
    coupling_weights: np.ndarray | None
    solver_status: str
    dephasing_target: np.ndarray


@dataclass
class ContextualFractionResult:
    """Result bundle for contextual-fraction assessment."""

    noncontextual_fraction: float
    contextual_fraction: float
    preparation_extremals: np.ndarray
    effect_extremals: np.ndarray
    coupling_weights: np.ndarray | None
    solver_status: str


def preparation_assignment_extremals(
    scenario: ContextualityScenario,
    atol: float | None = None,
) -> np.ndarray:
    """Enumerate extremal preparation-assignment rays via CDD.

    The preparation-assignment cone is defined over variables ``p(x,a)``:
    - ``p(x,a) >= 0`` for all ``x,a``
    - every preparation OPEQ holds pointwise:
      ``sum_{x,a} c[x,a] p(x,a) = 0``
    """
    tol = scenario.atol if atol is None else float(atol)
    rays_flat = _assignment_extremal_rays(
        opeq_array=scenario.opeq_preps_numeric,
        num_settings=scenario.X_cardinality,
        num_outcomes=scenario.A_cardinality,
        atol=tol,
    )
    rays = rays_flat.reshape(-1, scenario.X_cardinality, scenario.A_cardinality)
    _assert_zero_on_invalid_support(
        rays=rays,
        valid_mask=scenario.valid_a_mask,
        atol=tol,
        label="preparation assignment extremals",
    )
    return rays


def effect_assignment_extremals(
    scenario: ContextualityScenario,
    atol: float | None = None,
) -> np.ndarray:
    """Enumerate extremal effect-assignment rays via CDD.

    The effect-assignment cone is defined over variables ``q(y,b)``:
    - ``q(y,b) >= 0`` for all ``y,b``
    - every measurement OPEQ holds pointwise:
      ``sum_{y,b} d[y,b] q(y,b) = 0``
    """
    tol = scenario.atol if atol is None else float(atol)
    rays_flat = _assignment_extremal_rays(
        opeq_array=scenario.opeq_meas_numeric,
        num_settings=scenario.Y_cardinality,
        num_outcomes=scenario.B_cardinality,
        atol=tol,
    )
    rays = rays_flat.reshape(-1, scenario.Y_cardinality, scenario.B_cardinality)
    _assert_zero_on_invalid_support(
        rays=rays,
        valid_mask=scenario.valid_b_mask,
        atol=tol,
        label="effect assignment extremals",
    )
    return rays


def assess_simplex_embeddability(
    scenario: ContextualityScenario,
    dephasing_target: np.ndarray | None = None,
    atol: float | None = None,
) -> SimplexEmbeddabilityResult:
    """Assess simplex embeddability and dephasing robustness.

    Computes the minimum ``r in [0,1]`` such that the dephased behavior
    ``(1-r)P + r D`` admits a nonnegative assignment-ray decomposition.
    """
    tol = scenario.atol if atol is None else float(atol)
    data_numeric = scenario.data_numeric
    prep_extremals = preparation_assignment_extremals(scenario, atol=tol)
    effect_extremals = effect_assignment_extremals(scenario, atol=tol)
    target = (
        _default_dephasing_target(data_numeric, atol=tol)
        if dephasing_target is None
        else _validate_dephasing_target(
            np.asarray(dephasing_target, dtype=float),
            shape=data_numeric.shape,
            atol=tol,
        )
    )

    robustness, weights, status = _solve_dephasing_robustness_lp(
        data=data_numeric,
        dephasing_target=target,
        prep_extremals=prep_extremals,
        effect_extremals=effect_extremals,
        atol=tol,
    )
    return SimplexEmbeddabilityResult(
        is_simplex_embeddable=bool(np.isfinite(robustness) and robustness <= tol),
        dephasing_robustness=float(robustness),
        preparation_extremals=prep_extremals,
        effect_extremals=effect_extremals,
        coupling_weights=weights,
        solver_status=status,
        dephasing_target=target,
    )


def contextuality_robustness_to_dephasing(
    scenario: ContextualityScenario,
    dephasing_target: np.ndarray | None = None,
    atol: float | None = None,
) -> float:
    """Return only the contextuality measure: robustness to dephasing."""
    return assess_simplex_embeddability(
        scenario=scenario,
        dephasing_target=dephasing_target,
        atol=atol,
    ).dephasing_robustness


def assess_contextual_fraction(
    scenario: ContextualityScenario,
    atol: float | None = None,
) -> ContextualFractionResult:
    """Assess noncontextual/contextual fractions via a cone-subbehavior LP."""
    tol = scenario.atol if atol is None else float(atol)
    data_numeric = scenario.data_numeric
    prep_extremals = preparation_assignment_extremals(scenario, atol=tol)
    effect_extremals = effect_assignment_extremals(scenario, atol=tol)

    noncontextual, weights, status = _solve_noncontextual_fraction_lp(
        data=data_numeric,
        prep_extremals=prep_extremals,
        effect_extremals=effect_extremals,
        atol=tol,
    )
    if np.isfinite(noncontextual):
        if noncontextual < -10.0 * tol or noncontextual > 1.0 + 10.0 * tol:
            raise RuntimeError(
                "Solved noncontextual_fraction is outside [0, 1] beyond tolerance. "
                "This indicates a numerical/solver issue."
            )
        noncontextual = float(np.clip(noncontextual, 0.0, 1.0))
    contextual = (1.0 - noncontextual) if np.isfinite(noncontextual) else float("nan")

    return ContextualFractionResult(
        noncontextual_fraction=float(noncontextual),
        contextual_fraction=float(contextual),
        preparation_extremals=prep_extremals,
        effect_extremals=effect_extremals,
        coupling_weights=weights,
        solver_status=status,
    )


def noncontextual_fraction(
    scenario: ContextualityScenario,
    atol: float | None = None,
) -> float:
    """Return only the noncontextual fraction."""
    return assess_contextual_fraction(scenario=scenario, atol=atol).noncontextual_fraction


def contextual_fraction(
    scenario: ContextualityScenario,
    atol: float | None = None,
) -> float:
    """Return only the contextual fraction."""
    return assess_contextual_fraction(scenario=scenario, atol=atol).contextual_fraction


def _assignment_extremal_rays(
    opeq_array: np.ndarray,
    num_settings: int,
    num_outcomes: int,
    atol: float,
) -> np.ndarray:
    """Build assignment cone and return extremal rays as flat vectors."""
    num_vars = num_settings * num_outcomes
    opeq_rows = select_linearly_independent_rows(
        np.asarray(opeq_array, dtype=float).reshape(-1, num_vars),
        atol=atol,
        method="numpy",
    )
    return enumerate_cone_extremal_rays(opeq_rows, atol=atol, method="cdd")


def _assert_zero_on_invalid_support(
    rays: np.ndarray,
    valid_mask: np.ndarray,
    atol: float,
    label: str,
) -> None:
    """Sanity-check that extremal rays vanish on padded coordinates."""
    arr = np.asarray(rays, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool)
    if arr.ndim != 3:
        raise ValueError(f"{label}: rays must have shape (N, S, O).")
    if mask.shape != arr.shape[1:]:
        raise ValueError(f"{label}: valid_mask shape mismatch.")
    invalid = np.broadcast_to(~mask[np.newaxis, :, :], arr.shape)
    if invalid.any() and np.any(np.abs(arr[invalid]) > float(atol)):
        raise RuntimeError(f"{label} have nonzero entries on padded invalid coordinates.")


def _solve_dephasing_robustness_lp(
    data: np.ndarray,
    dephasing_target: np.ndarray,
    prep_extremals: np.ndarray,
    effect_extremals: np.ndarray,
    atol: float,
) -> tuple[float, np.ndarray | None, str]:
    num_prep_vertices, num_effect_vertices, coeff_weights, _ = _assignment_product_blocks(
        prep_extremals=prep_extremals,
        effect_extremals=effect_extremals,
        data_shape=data.shape,
        atol=atol,
    )
    if dephasing_target.shape != data.shape:
        raise ValueError("dephasing_target must match data shape.")

    num_weights = num_prep_vertices * num_effect_vertices
    num_vars = num_weights + 1  # +1 for r
    num_rows = coeff_weights.shape[0]

    coeff_r = -(
        dephasing_target - data
    ).transpose(0, 2, 1, 3).reshape(num_rows, 1)
    coeff_matrix = np.hstack([coeff_weights, coeff_r])
    coeff_matrix = np.where(np.abs(coeff_matrix) <= atol, 0.0, coeff_matrix)

    sparse = csc_matrix(coeff_matrix)
    aptrb = sparse.indptr[:-1].astype(np.int64, copy=False)
    aptre = sparse.indptr[1:].astype(np.int64, copy=False)
    asub = sparse.indices.astype(np.int32, copy=False)
    aval = sparse.data.astype(np.float64, copy=False)

    rhs = data.transpose(0, 2, 1, 3).reshape(num_rows).astype(np.float64, copy=False)
    c = np.zeros(num_vars, dtype=np.float64)
    c[-1] = 1.0

    bkc = [mosek.boundkey.fx] * num_rows
    blc = rhs
    buc = rhs

    bkx = [mosek.boundkey.lo] * num_weights + [mosek.boundkey.fr]
    blx = np.zeros(num_vars, dtype=np.float64)
    bux = np.full(num_vars, np.inf, dtype=np.float64)

    with mosek.Env() as env:
        with env.Task(0, 0) as task:
            task.inputdata(
                num_rows,
                num_vars,
                c,
                0.0,
                aptrb,
                aptre,
                asub,
                aval,
                bkc,
                blc,
                buc,
                bkx,
                blx,
                bux,
            )

            # Prefer basic/corner LP solutions for decomposition weights.
            task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.primal_simplex)
            task.putobjsense(mosek.objsense.minimize)
            task.optimize()

            xx, status = _extract_optimal_solution_vector(task, num_vars=num_vars, atol=atol)
            if xx is not None:
                robustness = float(xx[-1])
                weights = xx[:-1].reshape(num_prep_vertices, num_effect_vertices)
                return robustness, weights, status
            return float("inf"), None, status

    return float("inf"), None, "unknown"


def _solve_noncontextual_fraction_lp(
    data: np.ndarray,
    prep_extremals: np.ndarray,
    effect_extremals: np.ndarray,
    atol: float,
) -> tuple[float, np.ndarray | None, str]:
    (
        num_prep_vertices,
        num_effect_vertices,
        coeff_weights,
        mass_weights,
    ) = _assignment_product_blocks(
        prep_extremals=prep_extremals,
        effect_extremals=effect_extremals,
        data_shape=data.shape,
        atol=atol,
    )

    num_weights = num_prep_vertices * num_effect_vertices
    num_vars = num_weights + 1  # +1 for lambda
    num_rows_data = coeff_weights.shape[0]
    num_rows_mass = mass_weights.shape[0]
    num_rows = num_rows_data + num_rows_mass

    # Inequalities: S <= P
    coeff_data = np.hstack([coeff_weights, np.zeros((num_rows_data, 1), dtype=float)])
    # Equal mass per (x,y): sum_ab S - lambda = 0
    coeff_mass = np.hstack([mass_weights, -np.ones((num_rows_mass, 1), dtype=float)])
    coeff_matrix = np.vstack([coeff_data, coeff_mass])
    coeff_matrix = np.where(np.abs(coeff_matrix) <= atol, 0.0, coeff_matrix)

    sparse = csc_matrix(coeff_matrix)
    aptrb = sparse.indptr[:-1].astype(np.int64, copy=False)
    aptre = sparse.indptr[1:].astype(np.int64, copy=False)
    asub = sparse.indices.astype(np.int32, copy=False)
    aval = sparse.data.astype(np.float64, copy=False)

    rhs_data = data.transpose(0, 2, 1, 3).reshape(num_rows_data).astype(np.float64, copy=False)
    rhs_mass = np.zeros(num_rows_mass, dtype=np.float64)

    bkc = [mosek.boundkey.up] * num_rows_data + [mosek.boundkey.fx] * num_rows_mass
    blc = np.concatenate(
        [
            np.full(num_rows_data, -np.inf, dtype=np.float64),
            rhs_mass,
        ]
    )
    buc = np.concatenate([rhs_data, rhs_mass])

    c = np.zeros(num_vars, dtype=np.float64)
    c[-1] = 1.0

    bkx = [mosek.boundkey.lo] * num_weights + [mosek.boundkey.ra]
    blx = np.zeros(num_vars, dtype=np.float64)
    bux = np.full(num_vars, np.inf, dtype=np.float64)
    bux[-1] = 1.0

    with mosek.Env() as env:
        with env.Task(0, 0) as task:
            task.inputdata(
                num_rows,
                num_vars,
                c,
                0.0,
                aptrb,
                aptre,
                asub,
                aval,
                bkc,
                blc,
                buc,
                bkx,
                blx,
                bux,
            )

            # Prefer basic/corner LP solutions for decomposition weights.
            task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.primal_simplex)
            task.putobjsense(mosek.objsense.maximize)
            task.optimize()

            xx, status = _extract_optimal_solution_vector(task, num_vars=num_vars, atol=atol)
            if xx is not None:
                noncontextual = float(xx[-1])
                weights = xx[:-1].reshape(num_prep_vertices, num_effect_vertices)
                return noncontextual, weights, status
            return float("nan"), None, status

    return float("nan"), None, "unknown"


def _assignment_product_blocks(
    prep_extremals: np.ndarray,
    effect_extremals: np.ndarray,
    data_shape: tuple[int, int, int, int],
    atol: float,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Build ray-product coefficient blocks for data and per-(x,y) mass rows."""
    num_prep_vertices, num_x, num_a = prep_extremals.shape
    num_effect_vertices, num_y, num_b = effect_extremals.shape
    if data_shape != (num_x, num_y, num_a, num_b):
        raise ValueError("data shape is inconsistent with extremal assignment dimensions.")

    num_weights = num_prep_vertices * num_effect_vertices
    num_rows_data = num_x * num_y * num_a * num_b

    prep_flat = prep_extremals.reshape(num_prep_vertices, num_x * num_a)
    effect_flat = effect_extremals.reshape(num_effect_vertices, num_y * num_b)

    # Rows indexed by (x,a,y,b), matching existing robustness flattening.
    coeff_weights = (
        np.einsum("ip,jq->ijpq", prep_flat, effect_flat, optimize=True)
        .reshape(num_weights, num_rows_data)
        .T
    )

    prep_masses = prep_extremals.sum(axis=2)  # (N_prep, X)
    effect_masses = effect_extremals.sum(axis=2)  # (N_eff, Y)
    mass_weights = (
        np.einsum("ix,jy->ijxy", prep_masses, effect_masses, optimize=True)
        .reshape(num_weights, num_x * num_y)
        .T
    )

    coeff_weights = np.where(np.abs(coeff_weights) <= atol, 0.0, coeff_weights)
    mass_weights = np.where(np.abs(mass_weights) <= atol, 0.0, mass_weights)
    return num_prep_vertices, num_effect_vertices, coeff_weights, mass_weights


def _extract_optimal_solution_vector(
    task: mosek.Task,
    num_vars: int,
    atol: float,
) -> tuple[np.ndarray | None, str]:
    """Return an optimal primal vector if available, otherwise ``(None, status)``."""
    acceptable_statuses = {mosek.solsta.optimal}
    if hasattr(mosek.solsta, "integer_optimal"):
        acceptable_statuses.add(mosek.solsta.integer_optimal)

    final_status = "unknown"
    # Prefer a basic LP solution (corner point) when available.
    for soltype in (mosek.soltype.bas, mosek.soltype.itr):
        try:
            solsta = task.getsolsta(soltype)
        except mosek.Error:
            continue
        final_status = str(solsta)
        if solsta in acceptable_statuses:
            xx = np.zeros(num_vars, dtype=float)
            task.getxx(soltype, xx)
            xx = np.where(np.abs(xx) <= atol, 0.0, xx)
            return xx, str(solsta)
    return None, final_status


def _default_dephasing_target(data: np.ndarray, atol: float) -> np.ndarray:
    """Default dephasing target ``D`` built from data marginals.

    Uses ``D(a,b|x,y)=P(a|x) * Q(b|y)`` where:
    - ``P(a|x)`` is averaged over y from the input table,
    - ``Q(b|y)`` is averaged over x from ``P(b|x,y)``.
    """
    num_x, _num_y, _num_a, _num_b = data.shape
    p_a_given_x = data.sum(axis=3).mean(axis=1)
    q_b_given_y = data.sum(axis=(0, 2)) / float(num_x)
    p_a_given_x = _normalize_rows(p_a_given_x, atol=atol)
    q_b_given_y = _normalize_rows(q_b_given_y, atol=atol)
    return p_a_given_x[:, None, :, None] * q_b_given_y[None, :, None, :]


def _validate_dephasing_target(
    target: np.ndarray,
    shape: tuple[int, int, int, int],
    atol: float,
) -> np.ndarray:
    if target.shape != shape:
        raise ValueError(f"dephasing_target must have shape {shape}.")
    if np.any(target < -atol):
        raise ValueError("dephasing_target contains negative entries.")
    if not np.allclose(target.sum(axis=(2, 3)), 1.0, atol=atol):
        raise ValueError("Each (x,y) in dephasing_target must sum to 1 over (a,b).")
    return np.asarray(target, dtype=float)


def _normalize_rows(mat: np.ndarray, atol: float) -> np.ndarray:
    arr = np.asarray(mat, dtype=float)
    arr = np.where(np.abs(arr) <= atol, 0.0, arr)
    arr = np.maximum(arr, 0.0)
    row_sums = arr.sum(axis=1, keepdims=True)
    if np.any(row_sums <= atol):
        raise ValueError("Cannot normalize rows with zero total mass.")
    return arr / row_sums

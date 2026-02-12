"""Simplex-embeddability and contextuality robustness for scenarios.

This module works in three steps:
1. Enumerate extremal preparation/effect assignment rays with CDD.
2. Build a dephasing target behavior.
3. Solve one MOSEK LP for the minimum dephasing weight making the behavior
   representable by nonnegative combinations of assignment-ray products.
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
        opeq_array=scenario.opeq_preps,
        num_settings=scenario.X_cardinality,
        num_outcomes=scenario.A_cardinality,
        atol=tol,
    )
    return rays_flat.reshape(-1, scenario.X_cardinality, scenario.A_cardinality)


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
        opeq_array=scenario.opeq_meas,
        num_settings=scenario.Y_cardinality,
        num_outcomes=scenario.B_cardinality,
        atol=tol,
    )
    return rays_flat.reshape(-1, scenario.Y_cardinality, scenario.B_cardinality)


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
    prep_extremals = preparation_assignment_extremals(scenario, atol=tol)
    effect_extremals = effect_assignment_extremals(scenario, atol=tol)
    target = (
        _default_dephasing_target(scenario.data, atol=tol)
        if dephasing_target is None
        else _validate_dephasing_target(
            np.asarray(dephasing_target, dtype=float),
            shape=scenario.data.shape,
            atol=tol,
        )
    )

    robustness, weights, status = _solve_dephasing_robustness_lp(
        data=scenario.data,
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


def _solve_dephasing_robustness_lp(
    data: np.ndarray,
    dephasing_target: np.ndarray,
    prep_extremals: np.ndarray,
    effect_extremals: np.ndarray,
    atol: float,
) -> tuple[float, np.ndarray | None, str]:
    num_prep_vertices, num_x, num_a = prep_extremals.shape
    num_effect_vertices, num_y, num_b = effect_extremals.shape

    if data.shape != (num_x, num_y, num_a, num_b):
        raise ValueError("data shape is inconsistent with extremal assignment dimensions.")
    if dephasing_target.shape != data.shape:
        raise ValueError("dephasing_target must match data shape.")

    num_weights = num_prep_vertices * num_effect_vertices
    num_vars = num_weights + 1  # +1 for r
    num_rows = num_x * num_y * num_a * num_b

    prep_flat = prep_extremals.reshape(num_prep_vertices, num_x * num_a)
    effect_flat = effect_extremals.reshape(num_effect_vertices, num_y * num_b)

    # Rows are indexed by (x,a,y,b), enabling vectorized Khatri-Rao construction.
    coeff_weights = (
        np.einsum("ip,jq->ijpq", prep_flat, effect_flat, optimize=True)
        .reshape(num_weights, num_rows)
        .T
    )
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

            task.putobjsense(mosek.objsense.minimize)
            task.optimize()

            acceptable_statuses = {mosek.solsta.optimal}
            if hasattr(mosek.solsta, "integer_optimal"):
                acceptable_statuses.add(mosek.solsta.integer_optimal)

            final_status = "unknown"
            for soltype in (mosek.soltype.itr, mosek.soltype.bas):
                try:
                    solsta = task.getsolsta(soltype)
                except mosek.Error:
                    continue
                final_status = str(solsta)
                if solsta in acceptable_statuses:
                    xx = np.zeros(num_vars, dtype=float)
                    task.getxx(soltype, xx)
                    xx = np.where(np.abs(xx) <= atol, 0.0, xx)
                    robustness = float(xx[-1])
                    weights = xx[:-1].reshape(num_prep_vertices, num_effect_vertices)
                    return robustness, weights, str(solsta)
            return float("inf"), None, final_status

    return float("inf"), None, "unknown"


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

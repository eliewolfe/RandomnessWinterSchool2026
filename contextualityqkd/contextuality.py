"""Simplex-embeddability and contextuality quantifiers for scenarios.

A behavior ``P(b|x,y)`` is noncontextual iff it lies in the cone spanned by the
products of preparation- and effect-assignment extremal rays,
``S_ij(x,y,b) = prep_i(x) * effect_j(y,b)``. Two monotones quantify how far the
observed data sits from that cone, both as readable CVXPY LPs over a single
nonnegative weight array ``w[i, j]``:

1. **Contextual fraction** -- ``1 - lambda*`` where ``lambda*`` is the maximum
   uniform mass of a noncontextual subbehavior ``S <= P``.
2. **Dephasing robustness** -- the minimum ``r`` such that ``(1-r)P + r D`` is
   noncontextual (``D`` a dephasing target).

Each LP's dual yields a **noncontextuality inequality** (a separating
hyperplane) that the observed data violates by exactly the monotone value, so a
zero violation certifies simplex-embeddability.
"""

from __future__ import annotations

from functools import cached_property
from typing import Literal

import cvxpy as cp
import numpy as np

from .cvxpy_utils import (
    NamedDuals,
    assert_cvxpy_solution_is_optimal,
    build_solve_kwargs,
    resolve_backend_solver,
    uses_gurobi_simplex,
    uses_mosek_simplex,
)
from .linalg_utils import enumerate_cone_extremal_rays, select_linearly_independent_rows
from .scenario import ContextualityScenario


Monotone = Literal["contextual_fraction", "dephasing_robustness"]
_MONOTONES: tuple[Monotone, ...] = ("contextual_fraction", "dephasing_robustness")


class NoncontextualityAssessment:
    """Contextuality monotones plus dual noncontextuality-inequality witnesses.

    A single class computes either or both monotones for ``scenario`` over a
    shared nonnegative weight variable ``w`` and shared subbehavior expression
    ``S = sum_ij w_ij prep_i (x) effect_j``. Numeric results and the witness
    inequality/violation for each monotone are exposed as cached properties.
    """

    def __init__(
        self,
        scenario: ContextualityScenario,
        *,
        monotone: Monotone | Literal["both"] = "contextual_fraction",
        dephasing_target: np.ndarray | None = None,
        atol: float | None = None,
        backend_solver: str = "highs",
        qp_solver: str = "gurobi",
        threads: int | None = None,
        verbose: int | bool = 0,
    ) -> None:
        if not isinstance(scenario, ContextualityScenario):
            raise TypeError("scenario must be a ContextualityScenario instance.")
        self.scenario = scenario
        self.num_x = int(scenario.X_cardinality)
        self.num_y = int(scenario.Y_cardinality)
        self.num_b = int(scenario.B_cardinality)
        self.atol = scenario.atol if atol is None else float(atol)
        self.backend_solver = str(backend_solver).strip().lower()
        self.solver = resolve_backend_solver(self.backend_solver)
        self._mosek_simplex = uses_mosek_simplex(self.backend_solver)
        self._gurobi_simplex = uses_gurobi_simplex(self.backend_solver)
        self.qp_solver = str(qp_solver).strip().lower()
        self.threads = None if threads is None else int(threads)
        self.verbose = int(verbose)
        self.requested_monotones = self._normalize_monotone(monotone)
        self._dephasing_target_arg = dephasing_target
        # _solved: primal-side info per monotone (measure, weights, subbehavior,
        # raw LP dual snapshot, primal_value). _refined: witness-side info per
        # monotone built lazily by _ensure_refined (alpha, bound, sense,
        # violation, dual_values with QP-refined witness duals).
        self._solved: dict[Monotone, dict[str, object]] = {}
        self._refined: dict[Monotone, dict[str, object]] = {}

    def solve(self) -> "NoncontextualityAssessment":
        """Eagerly solve every requested monotone; return self for chaining."""
        for monotone in self.requested_monotones:
            self._ensure_solved(monotone)
        return self

    # ----- numeric results (each solves its monotone lazily, then caches) -----

    @cached_property
    def noncontextual_fraction(self) -> float:
        """Maximum uniform mass ``lambda*`` of a noncontextual subbehavior."""
        return float(self._ensure_solved("contextual_fraction")["noncontextual_fraction"])

    @cached_property
    def contextual_fraction(self) -> float:
        """``1 - noncontextual_fraction``."""
        return float(self._ensure_solved("contextual_fraction")["measure"])

    @cached_property
    def dephasing_robustness(self) -> float:
        """Minimum dephasing ``r*`` to enter the noncontextual cone (may be < 0)."""
        return float(self._ensure_solved("dephasing_robustness")["measure"])

    @cached_property
    def is_simplex_embeddable(self) -> bool:
        """True iff the behavior is already noncontextual (``r* <= atol``)."""
        return bool(self.dephasing_robustness <= self.atol)

    @cached_property
    def contextual(self) -> bool:
        """True iff the requested monotone certifies contextuality (measure > atol)."""
        if "contextual_fraction" in self.requested_monotones:
            return bool(self.contextual_fraction > self.atol)
        return bool(self.dephasing_robustness > self.atol)

    @cached_property
    def dephasing_target(self) -> np.ndarray:
        """The dephasing target ``D`` used by the robustness LP."""
        return self._resolve_dephasing_target()

    # ----- per-monotone bundles (dict keyed by requested monotone) -----

    @cached_property
    def coupling_weights(self) -> dict[Monotone, np.ndarray]:
        """Decomposition weights ``w[i, j]`` per requested monotone."""
        return {m: self._ensure_solved(m)["weights"] for m in self.requested_monotones}

    @cached_property
    def subbehavior(self) -> dict[Monotone, np.ndarray]:
        """Reconstructed (sub)behavior ``S(x, y, b)`` per requested monotone."""
        return {m: self._ensure_solved(m)["subbehavior"] for m in self.requested_monotones}

    @cached_property
    def dual_values(self) -> dict[Monotone, dict[str, np.ndarray | None]]:
        """QP-refined constraint duals per requested monotone.

        Witness-relevant duals (``subbehavior_le_data``+``uniform_mass`` for
        contextual fraction; ``dephased_behavior`` for dephasing robustness)
        are the minimum-norm point on the optimal dual face. Auxiliary duals
        not in the QP objective (the implicit ``lambda <= 1`` multiplier) are
        not surfaced here. For the raw LP-side snapshot, see
        :attr:`raw_dual_values`.
        """
        return {m: self._ensure_refined(m)["dual_values"] for m in self.requested_monotones}

    @cached_property
    def raw_dual_values(self) -> dict[Monotone, dict[str, np.ndarray | None]]:
        """Raw LP-side constraint duals per requested monotone, before QP refinement."""
        return {m: self._ensure_solved(m)["raw_duals"] for m in self.requested_monotones}

    # ----- dual noncontextuality-inequality witnesses (all QP-refined) -----

    @cached_property
    def inequality(self) -> dict[Monotone, np.ndarray]:
        """Witness functional ``alpha(x, y, b)`` (shape of ``P``) per monotone."""
        return {m: self._ensure_refined(m)["alpha"] for m in self.requested_monotones}

    @cached_property
    def inequality_bound(self) -> dict[Monotone, float]:
        """Noncontextual bound ``beta`` per monotone (``<alpha, N> sense beta``)."""
        return {m: float(self._ensure_refined(m)["bound"]) for m in self.requested_monotones}

    @cached_property
    def inequality_sense(self) -> dict[Monotone, str]:
        """Sense ``"<="`` or ``">="`` satisfied by all noncontextual behaviors."""
        return {m: str(self._ensure_refined(m)["sense"]) for m in self.requested_monotones}

    @cached_property
    def violation(self) -> dict[Monotone, float]:
        """Signed amount the observed data breaks the inequality (== measure)."""
        return {m: float(self._ensure_refined(m)["violation"]) for m in self.requested_monotones}

    # ----- shared model scaffolding -----

    @cached_property
    def prep_extremals(self) -> np.ndarray:
        """Extremal preparation-assignment rays, shape ``(N_prep, num_x)``."""
        return preparation_assignment_extremals(self.scenario, atol=self.atol)

    @cached_property
    def effect_extremals(self) -> np.ndarray:
        """Extremal effect-assignment rays, shape ``(N_eff, num_y, num_b)``."""
        return effect_assignment_extremals(self.scenario, atol=self.atol)

    @cached_property
    def _weights(self) -> cp.Variable:
        n_prep = self.prep_extremals.shape[0]
        n_eff = self.effect_extremals.shape[0]
        return cp.Variable((n_prep, n_eff), nonneg=True, name="w")

    @cached_property
    def _subbehavior_expr(self) -> cp.Expression:
        """``S(x, y, b) = sum_ij w_ij prep_i(x) effect_j(y, b)`` via two matmuls."""
        prep = self.prep_extremals                       # (N_prep, num_x)
        effect_flat = self.effect_extremals.reshape(self.effect_extremals.shape[0], -1)  # C-order
        effect_mix = self._weights @ effect_flat         # (N_prep, num_y*num_b)
        flat = prep.T @ effect_mix                       # (num_x, num_y*num_b)
        return cp.reshape(flat, (self.num_x, self.num_y, self.num_b), order="C")

    def _build_contextual_fraction(self) -> dict[str, object]:
        """max lambda  s.t.  S <= P,  sum_b S == lambda,  0 <= lambda <= 1."""
        data = np.asarray(self.scenario.data_numeric, dtype=float)
        S = self._subbehavior_expr
        lam = cp.Variable(name="lambda")
        duals = NamedDuals()
        duals.add("subbehavior_le_data", S <= data)          # mu >= 0
        duals.add("uniform_mass", cp.sum(S, axis=2) == lam)  # nu; per-(x,y) mass equals lambda
        problem = cp.Problem(cp.Maximize(lam), duals.constraints + [lam <= 1.0])  # S, lambda >= 0 implied
        return {"problem": problem, "duals": duals}

    def _build_dephasing_robustness(self) -> dict[str, object]:
        """min r  s.t.  S == (1 - r) P + r D,  r free."""
        data = np.asarray(self.scenario.data_numeric, dtype=float)
        target = self._resolve_dephasing_target()
        S = self._subbehavior_expr
        r = cp.Variable(name="r")
        duals = NamedDuals()
        duals.add("dephased_behavior", S == (1.0 - r) * data + r * target)  # witness functional
        problem = cp.Problem(cp.Minimize(r), duals.constraints)
        return {"problem": problem, "duals": duals}

    def _ensure_solved(self, monotone: Monotone) -> dict[str, object]:
        """Solve the LP for ``monotone`` once; cache only primal-side quantities."""
        monotone = self._canonical_monotone(monotone)
        if monotone in self._solved:
            return self._solved[monotone]

        spec = (
            self._build_contextual_fraction()
            if monotone == "contextual_fraction"
            else self._build_dephasing_robustness()
        )
        problem: cp.Problem = spec["problem"]  # type: ignore[assignment]
        duals: NamedDuals = spec["duals"]      # type: ignore[assignment]

        # These are pure LPs (Zero + NonNeg cones); CVXPY returns all duals
        # directly, so no special dual-reshape workaround is needed.
        value = float(problem.solve(**self._solve_kwargs()))
        assert_cvxpy_solution_is_optimal(problem=problem, value=value, problem_kind=f"{monotone} LP")

        weights = np.asarray(self._weights.value, dtype=float)
        subbehavior = np.asarray(self._subbehavior_expr.value, dtype=float)
        raw_duals = duals.snapshot()

        if monotone == "contextual_fraction":
            noncontextual = self._validated_noncontextual_fraction(value)
            measure = 1.0 - noncontextual
            extra: dict[str, object] = {"noncontextual_fraction": float(noncontextual)}
        else:
            measure = float(value)  # r*
            extra = {}

        solution: dict[str, object] = {
            "measure": float(measure),
            "weights": weights,
            "subbehavior": subbehavior,
            "raw_duals": raw_duals,
            "primal_value": float(value),
            **extra,
        }
        self._solved[monotone] = solution
        return solution

    def _ensure_refined(self, monotone: Monotone) -> dict[str, object]:
        """Run the L2 dual-refinement QP for ``monotone`` once; cache the witness."""
        from .dual_refinement import (
            build_dual_contextual_fraction,
            build_dual_dephasing_robustness,
        )

        monotone = self._canonical_monotone(monotone)
        if monotone in self._refined:
            return self._refined[monotone]
        primal = self._ensure_solved(monotone)
        data = np.asarray(self.scenario.data_numeric, dtype=float)
        primal_value = float(primal["primal_value"])  # type: ignore[arg-type]

        if monotone == "contextual_fraction":
            refined_duals = build_dual_contextual_fraction(
                data=data,
                prep_extremals=self.prep_extremals,
                effect_extremals=self.effect_extremals,
                primal_value=primal_value,
                qp_solver=self.qp_solver,
                threads=self.threads,
                verbose=self.verbose >= 2,
            )
            mu = refined_duals["subbehavior_le_data"]
            nu = refined_duals["uniform_mass"]
            alpha = -(mu + nu[:, :, None])
        else:
            refined_duals = build_dual_dephasing_robustness(
                data=data,
                target=self._resolve_dephasing_target(),
                prep_extremals=self.prep_extremals,
                effect_extremals=self.effect_extremals,
                primal_value=primal_value,
                qp_solver=self.qp_solver,
                threads=self.threads,
                verbose=self.verbose >= 2,
            )
            # CVXPY's equality-constraint dual returns gamma = -alpha_witness;
            # the QP keeps that convention so dual_values matches the LP-side
            # sign and existing downstream sign flips stay correct.
            alpha = -refined_duals["dephased_behavior"]

        bound = 0.0
        sense = "<="
        violation = float(np.sum(alpha * data)) - bound

        refined: dict[str, object] = {
            "alpha": np.asarray(alpha, dtype=float),
            "bound": float(bound),
            "sense": sense,
            "violation": float(violation),
            "dual_values": refined_duals,
        }
        self._refined[monotone] = refined
        return refined

    def _validated_noncontextual_fraction(self, value: float) -> float:
        if not np.isfinite(value):
            return float("nan")
        if value < -10.0 * self.atol or value > 1.0 + 10.0 * self.atol:
            raise RuntimeError(
                "Solved noncontextual_fraction is outside [0, 1] beyond tolerance. "
                "This indicates a numerical/solver issue."
            )
        return float(np.clip(value, 0.0, 1.0))

    def _resolve_dephasing_target(self) -> np.ndarray:
        data = np.asarray(self.scenario.data_numeric, dtype=float)
        if self._dephasing_target_arg is None:
            return _default_dephasing_target(data, atol=self.atol)
        return _validate_dephasing_target(
            np.asarray(self._dephasing_target_arg, dtype=float),
            shape=data.shape,
            atol=self.atol,
        )

    def _solve_kwargs(self) -> dict[str, object]:
        return build_solve_kwargs(
            self.solver,
            mosek_simplex=self._mosek_simplex,
            gurobi_simplex=self._gurobi_simplex,
            threads=self.threads,
            verbose=self.verbose >= 2,
        )

    @staticmethod
    def _normalize_monotone(monotone: str) -> tuple[Monotone, ...]:
        token = str(monotone).strip().lower()
        if token == "both":
            return _MONOTONES
        return (NoncontextualityAssessment._canonical_monotone(token),)

    @staticmethod
    def _canonical_monotone(monotone: str) -> Monotone:
        token = str(monotone).strip().lower()
        if token in _MONOTONES:
            return token  # type: ignore[return-value]
        raise ValueError(
            f"monotone must be one of {_MONOTONES} or 'both', got {monotone!r}."
        )


def preparation_assignment_extremals(
    scenario: ContextualityScenario,
    atol: float | None = None,
) -> np.ndarray:
    """Enumerate extremal preparation-assignment rays via CDD.

    The preparation-assignment cone is defined over variables ``p(x)``:
    - ``p(x) >= 0`` for all ``x``
    - every preparation OPEQ holds pointwise: ``sum_x c[x] p(x) = 0``
    """
    tol = scenario.atol if atol is None else float(atol)
    rays_flat = _assignment_extremal_rays(
        opeq_array=scenario.opeq_preps_numeric,
        num_settings=scenario.X_cardinality,
        num_outcomes=1,
        atol=tol,
    )
    return rays_flat.reshape(-1, scenario.X_cardinality)


def effect_assignment_extremals(
    scenario: ContextualityScenario,
    atol: float | None = None,
) -> np.ndarray:
    """Enumerate extremal effect-assignment rays via CDD.

    The effect-assignment cone is defined over variables ``q(y,b)``:
    - ``q(y,b) >= 0`` for all ``y,b``
    - every measurement OPEQ holds pointwise: ``sum_{y,b} d[y,b] q(y,b) = 0``
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


def _default_dephasing_target(data: np.ndarray, atol: float) -> np.ndarray:
    """Default dephasing target ``D`` built from data marginals.

    Uses ``D(b|x,y)=Q(b|y)`` where ``Q(b|y)`` is averaged over x from ``p(b|x,y)``.
    """
    num_x, _num_y, _num_b = data.shape
    q_b_given_y = data.sum(axis=0) / float(num_x)
    q_b_given_y = _normalize_rows(q_b_given_y, atol=atol)
    return np.broadcast_to(q_b_given_y[np.newaxis, :, :], data.shape).copy()


def _validate_dephasing_target(
    target: np.ndarray,
    shape: tuple[int, int, int],
    atol: float,
) -> np.ndarray:
    if target.shape != shape:
        raise ValueError(f"dephasing_target must have shape {shape}.")
    if np.any(target < -atol):
        raise ValueError("dephasing_target contains negative entries.")
    if not np.allclose(target.sum(axis=2), 1.0, atol=atol):
        raise ValueError("Each (x,y) in dephasing_target must sum to 1 over b.")
    return np.asarray(target, dtype=float)


def _normalize_rows(mat: np.ndarray, atol: float) -> np.ndarray:
    arr = np.asarray(mat, dtype=float)
    arr = np.where(np.abs(arr) <= atol, 0.0, arr)
    arr = np.maximum(arr, 0.0)
    row_sums = arr.sum(axis=1, keepdims=True)
    if np.any(row_sums <= atol):
        raise ValueError("Cannot normalize rows with zero total mass.")
    return arr / row_sums


__all__ = [
    "NoncontextualityAssessment",
    "preparation_assignment_extremals",
    "effect_assignment_extremals",
]

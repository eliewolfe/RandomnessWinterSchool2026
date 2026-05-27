"""Readable CVXPY LP backend for contextuality-QKD Eve bounds.

The decision variable is the joint distribution ``P[x, y, b, e]`` -- the
probability that Bob outputs ``b`` and Eve guesses ``e`` given Alice's
preparation ``x`` and Bob's setting ``y``. It is modelled directly as a 4-D
:class:`cvxpy.Variable`, so every constraint reads like the math it encodes
(e.g. ``cp.sum(P, axis=3) == data`` is "summing over Eve's guess returns the
observed Bob marginal"). No flat-index arithmetic is needed.

Performance notes
-----------------
The LP is solved once per Bob setting ``y``; every ``y`` shares the same
feasible region and only the linear objective changes. Two choices keep this
fast without sacrificing readability:

1. **Parameterized objective.** The objective is
   ``cp.sum(cp.multiply(objective_coeffs, P))`` with ``objective_coeffs`` a
   :class:`cvxpy.Parameter`. This is DPP-compliant, so CVXPY canonicalizes the
   model exactly once and, for each ``y``, only swaps the cost tensor before
   re-solving -- recovering the old "build once / hot-swap objective" behavior.

2. **Vectorized constraints.** Each constraint group is a single tensor
   constraint over whole axes of ``P`` (a handful of constraints total) rather
   than thousands of scalar ones, so canonicalization stays cheap. Linear
   dependency detection / redundant-row elimination is left to MOSEK's presolve.
"""

from __future__ import annotations

from typing import Literal, Sequence

import cvxpy as cp
import numpy as np

from .cvxpy_utils import (
    NamedDuals,
    assert_cvxpy_solution_is_optimal,
    build_solve_kwargs,
    resolve_backend_solver,
    uses_mosek_simplex,
)
from .scenario import ContextualityScenario


class QKDNoncontextualLP:
    """Single-model noncontextual LP for Eve's QKD guessing attack."""

    def __init__(
        self,
        scenario: ContextualityScenario,
        *,
        master_key_holder: Literal["Alice", "Bob"] | str = "Alice",
        where_key: Sequence[Sequence[int]] | None = None,
        backend_solver: str = "mosek_simplex",
        threads: int | None = None,
        atol: float | None = None,
        verbose: int | bool = 0,
    ) -> None:
        if not isinstance(scenario, ContextualityScenario):
            raise TypeError("scenario must be a ContextualityScenario instance.")
        self.scenario = scenario
        self.num_x = int(scenario.X_cardinality)
        self.num_y = int(scenario.Y_cardinality)
        self.num_b = int(scenario.B_cardinality)
        self.b_cardinality_per_y = scenario.b_cardinality_per_y.astype(int, copy=False)
        self.num_e = self.num_b
        self.shape = (self.num_x, self.num_y, self.num_b, self.num_e)
        self.master_key_holder = self._canonicalize_master_key_holder(master_key_holder)
        self.where_key = self._normalize_where_key(where_key)
        self.backend_solver = str(backend_solver).strip().lower()
        self.solver = resolve_backend_solver(self.backend_solver)
        self._mosek_simplex = uses_mosek_simplex(self.backend_solver)
        self.threads = None if threads is None else int(threads)
        self.atol = scenario.atol if atol is None else float(atol)
        self.verbose = int(verbose)

        self.cvxpy_variable: cp.Variable | None = None
        self.cvxpy_constraints: list[cp.Constraint] = []
        self.objective_coeffs: cp.Parameter | None = None
        self.objective_vectors_by_y: dict[int, np.ndarray] = {}
        self.cvxpy_problem: cp.Problem | None = None
        self.cvxpy_problems_by_y: dict[int, cp.Problem] = {}
        self.duals = NamedDuals()
        self.dual_values_by_y: dict[int, dict[str, np.ndarray | None]] = {}
        self.eve_guess_by_y: np.ndarray | None = None
        self.solution_probabilities: np.ndarray | None = None

    def solve_lp(
        self,
        where_key: Sequence[Sequence[int]] | None = None,
        *,
        master_key_holder: Literal["Alice", "Bob"] | str | None = None,
        backend_solver: str | None = None,
    ) -> np.ndarray:
        """Solve one Eve guessing LP per Bob setting y."""
        if where_key is not None:
            self.where_key = self._normalize_where_key(where_key)
        if master_key_holder is not None:
            self.master_key_holder = self._canonicalize_master_key_holder(master_key_holder)
        if backend_solver is not None:
            self.backend_solver = str(backend_solver).strip().lower()
            self.solver = resolve_backend_solver(self.backend_solver)
            self._mosek_simplex = uses_mosek_simplex(self.backend_solver)

        self._build_problem()
        out = np.full((self.num_y,), np.nan, dtype=float)
        self.cvxpy_problems_by_y = {}
        self.dual_values_by_y = {}

        problem = self.cvxpy_problem
        assert problem is not None and self.objective_coeffs is not None
        solve_kwargs = self._solve_kwargs()
        for y in range(self.num_y):
            if self._where_key_row(y).size == 0:
                continue

            # Re-use the single compiled problem; only the cost tensor changes.
            # DPP keeps the canonicalization cached, so this forwards just a new
            # objective to the solver rather than rebuilding the model.
            self.objective_coeffs.value = self.objective_vectors_by_y[y]
            value = float(problem.solve(**solve_kwargs))
            assert_cvxpy_solution_is_optimal(problem=problem, value=value, problem_kind=f"LP for y={y}")
            out[y] = float(value)
            self.cvxpy_problems_by_y[y] = problem
            self.dual_values_by_y[y] = self.duals.snapshot()
            self._verify_guess_bound_tight(y, out[y])

        self.eve_guess_by_y = out
        self.solution_probabilities = self._extract_solution_probabilities()
        return out.copy()

    def _verify_guess_bound_tight(self, y: int, primal_value: float) -> None:
        """Assert the dual witness reproduces the primal guessing probability.

        The only nonzero-rhs constraint is ``observed_bob`` (rhs = data), so by
        strong duality ``<c_y, data>`` must equal the primal optimum ``G(y)``.
        This guards the inequality ``P_guess(y) <= <c_y, P(b|x,y)>`` against any
        solver-dependent dual sign/convention surprise.
        """
        coeffs = self.dual_values_by_y[y].get("observed_bob")
        if coeffs is None:
            raise RuntimeError(f"Missing observed_bob dual for y={y}; cannot form the guessing bound.")
        data = np.asarray(self.scenario.data_numeric, dtype=float)
        bound = float(np.sum(np.asarray(coeffs, dtype=float) * data))
        tol = max(1e-6, 1e-5 * abs(primal_value))
        if not np.isfinite(bound) or abs(bound - primal_value) > tol:
            raise RuntimeError(
                f"Eve guessing-bound dual inconsistency at y={y}: primal G={primal_value:.6g} "
                f"but <c_y, data>={bound:.6g} (|diff| > {tol:.1e}). The dual witness is not tight."
            )

    def solve(
        self,
        where_key: Sequence[Sequence[int]] | None = None,
        *,
        master_key_holder: Literal["Alice", "Bob"] | str | None = None,
        backend_solver: str | None = None,
    ) -> np.ndarray:
        return self.solve_lp(
            where_key=where_key, master_key_holder=master_key_holder, backend_solver=backend_solver
        )

    def _build_problem(self) -> None:
        self.cvxpy_variable = cp.Variable(self.shape, name="P")
        self.cvxpy_problem = None
        self.cvxpy_problems_by_y = {}
        self.duals = NamedDuals()
        self.dual_values_by_y = {}
        self.objective_vectors_by_y = {}
        self.solution_probabilities = None

        self._add_constraints()
        self.cvxpy_constraints = self.duals.constraints

        # Parameterized objective: build the model ONCE, then re-solve per y by
        # only assigning `objective_coeffs.value`. `sum(coeffs * P)` is
        # DPP-compliant (a parameter-affine objective), so CVXPY caches the
        # canonicalization across solves and only ships a fresh cost tensor.
        self.objective_coeffs = cp.Parameter(self.shape, name="objective_coeffs")
        self.cvxpy_problem = cp.Problem(
            cp.Maximize(cp.sum(cp.multiply(self.objective_coeffs, self.cvxpy_variable))),
            self.cvxpy_constraints,
        )
        self._build_objective_vectors()

    def _add_constraints(self) -> None:
        """Register every constraint group; each reports an array-valued dual."""
        P = self.cvxpy_variable
        preps = np.asarray(self.scenario.opeq_preps_numeric, dtype=float)   # (N_prep, num_x)
        meas = np.asarray(self.scenario.opeq_meas_numeric, dtype=float)     # (N_meas, num_y, num_b)
        data = np.asarray(self.scenario.data_numeric, dtype=float)          # (num_x, num_y, num_b)

        # P >= 0; dual is the reduced-cost tensor, shape (num_x, num_y, num_b, num_e).
        self.duals.add("probability_nonnegative", P >= 0.0)

        # Data consistency: sum_e P[x, y, b, e] = p_data(b | x, y).
        self.duals.add("observed_bob", cp.sum(P, axis=3) == data)

        # Preparation equivalence m: sum_x c[x] P[x, y, b, e] = 0 for every (y, b, e).
        # Broadcast c over (y, b, e), then contract the preparation axis 0.
        if preps.size:
            self.duals.add_group(
                "prep_opeq",
                [cp.sum(cp.multiply(c[:, None, None, None], P), axis=0) == 0.0 for c in preps],
            )

        # Measurement equivalence m: sum_{y,b} d[y,b] P[x, y, b, e] = 0 for every (x, e).
        # Broadcast d over (x, e), then contract the (y, b) measurement axes.
        if meas.size:
            self.duals.add_group(
                "measurement_opeq",
                [cp.sum(cp.multiply(d[None, :, :, None], P), axis=(1, 2)) == 0.0 for d in meas],
            )

        # Invalid guesses: force zero mass on e >= b_count(y). One masked constraint.
        mask = self._invalid_guess_mask()
        if mask.any():
            self.duals.add_masked("invalid_guess", P[mask] == 0.0, mask)

    def _invalid_guess_mask(self) -> np.ndarray:
        """Boolean (num_x, num_y, num_b, num_e) mask of forced-zero guess labels."""
        mask = np.zeros(self.shape, dtype=bool)
        for x, y in np.ndindex(self.num_x, self.num_y):
            b_count = int(self.b_cardinality_per_y[y])
            mask[x, y, :b_count, b_count:] = True  # valid b (< b_count), invalid e (>= b_count)
        return mask

    def _build_objective_vectors(self) -> None:
        """Precompute the per-y cost tensor assigned to ``objective_coeffs``."""
        key_lookup = self.scenario.key_selection_by_xy
        for y in range(self.num_y):
            vec = np.zeros(self.shape, dtype=float)
            x_row = self._where_key_row(y)
            if x_row.size == 0:
                self.objective_vectors_by_y[y] = vec
                continue

            weight = 1.0 / float(x_row.size)
            b_count = int(self.b_cardinality_per_y[y])
            for x in x_row.tolist():
                for b in range(b_count):
                    if self.master_key_holder == "Alice":
                        e = int(key_lookup[int(x), y])
                    else:
                        e = b
                    vec[int(x), y, b, e] += weight
            self.objective_vectors_by_y[y] = vec

    def _where_key_row(self, y: int) -> np.ndarray:
        return np.asarray(self.where_key[int(y)], dtype=int).reshape(-1)

    def _solve_kwargs(self) -> dict[str, object]:
        return build_solve_kwargs(
            self.solver,
            mosek_simplex=self._mosek_simplex,
            threads=self.threads,
            verbose=self.verbose >= 2,
        )

    def guess_bound_coeffs_by_y(self) -> dict[int, np.ndarray]:
        """Per-y witness coefficients c_y(x,y',b) for the guessing upper bound.

        For each solved y, the dual of the data-consistency constraint gives a
        linear upper bound on Eve's guessing probability in terms of the
        observed marginals: ``P_guess(y) <= sum_{x,y',b} c_y(x,y',b) P(b|x,y')``,
        tight at the observed data.
        """
        return {
            y: np.asarray(snapshot["observed_bob"], dtype=float)
            for y, snapshot in self.dual_values_by_y.items()
            if snapshot.get("observed_bob") is not None
        }

    def _extract_solution_probabilities(self) -> np.ndarray | None:
        if self.cvxpy_variable is None or self.cvxpy_variable.value is None:
            return None
        return np.asarray(self.cvxpy_variable.value, dtype=float)

    def _normalize_where_key(self, where_key: Sequence[Sequence[int]] | None) -> tuple[tuple[int, ...], ...]:
        if where_key is None:
            return tuple(tuple(range(self.num_x)) for _ in range(self.num_y))
        if len(where_key) != self.num_y:
            raise ValueError(f"where_key must have one row per y (expected {self.num_y}).")

        rows: list[tuple[int, ...]] = []
        for y, row in enumerate(where_key):
            arr = np.asarray(row, dtype=int).reshape(-1)
            if np.any(arr < 0) or np.any(arr >= self.num_x):
                raise ValueError(f"where_key[{y}] contains out-of-range x index.")
            rows.append(tuple(sorted(set(int(x) for x in arr.tolist()))))
        return tuple(rows)

    @staticmethod
    def _canonicalize_master_key_holder(master_key_holder: Literal["Alice", "Bob"] | str) -> Literal["Alice", "Bob"]:
        token = str(master_key_holder).strip().lower()
        if token == "alice":
            return "Alice"
        if token == "bob":
            return "Bob"
        raise ValueError("master_key_holder must be 'Alice' or 'Bob'.")


__all__ = ["QKDNoncontextualLP"]

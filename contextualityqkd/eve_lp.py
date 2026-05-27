"""Readable CVXPY LP backend for contextuality-QKD Eve bounds.

Performance notes
-----------------
The LP that bounds Eve's guessing probability is solved once per Bob setting
``y``. Every ``y`` shares the *same* feasible region (data-consistency,
operational-equivalence and structural-zero constraints); only the linear
objective changes. Two design choices keep this fast:

1. **Parameterized objective.** The objective is ``objective_coeffs @ P`` with
   ``objective_coeffs`` a :class:`cvxpy.Parameter`. This is DPP-compliant, so
   CVXPY canonicalizes (``ConeMatrixStuffing``) the model exactly once and, for
   each ``y``, only swaps the cost vector before re-solving. That restores the
   old raw-MOSEK "build once / hot-swap objective" behavior: canonicalization
   cost is paid a single time instead of ``num_y`` times.

2. **Stacked sparse constraints.** Equalities are assembled as a few
   ``A @ P == b`` blocks built from :mod:`scipy.sparse` matrices rather than
   thousands of scalar CVXPY constraints. CVXPY's matrix stuffing then has
   almost nothing to do (the model is already affine in matrix form), and any
   linear-dependency detection / redundant-row elimination is left to MOSEK's
   presolve where it belongs.
"""

from __future__ import annotations

from typing import Literal, Sequence

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from .cvxpy_utils import (
    DualKey,
    DualRef,
    add_named_constraint,
    assert_cvxpy_solution_is_optimal,
    is_mosek_solver,
    register_matrix_constraint,
    snapshot_dual_values,
    solve_cvxpy_problem_preserving_duals,
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
        solver: str | None = None,
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
        self.num_variables = self.num_x * self.num_y * self.num_b * self.num_e
        self.master_key_holder = self._canonicalize_master_key_holder(master_key_holder)
        self.where_key = self._normalize_where_key(where_key)
        self.solver = cp.MOSEK if solver is None else solver
        self.threads = None if threads is None else int(threads)
        self.atol = scenario.atol if atol is None else float(atol)
        self.verbose = int(verbose)

        self.cvxpy_variable: cp.Variable | None = None
        self.cvxpy_constraints: list[cp.Constraint] = []
        self.objective_coeffs: cp.Parameter | None = None
        self.objective_vectors_by_y: dict[int, np.ndarray] = {}
        self.cvxpy_problem: cp.Problem | None = None
        self.cvxpy_problems_by_y: dict[int, cp.Problem] = {}
        self.dual_constraints: dict[DualKey, DualRef] = {}
        self.dual_values_by_y: dict[int, dict[DualKey, object | None]] = {}
        self.eve_guess_by_y: np.ndarray | None = None
        self.solution_probabilities: np.ndarray | None = None

    def solve_lp(
        self,
        where_key: Sequence[Sequence[int]] | None = None,
        *,
        master_key_holder: Literal["Alice", "Bob"] | str | None = None,
        solver: str | None = None,
    ) -> np.ndarray:
        """Solve one Eve guessing LP per Bob setting y."""
        if where_key is not None:
            self.where_key = self._normalize_where_key(where_key)
        if master_key_holder is not None:
            self.master_key_holder = self._canonicalize_master_key_holder(master_key_holder)
        if solver is not None:
            self.solver = solver

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

            # Re-use the single compiled problem; only the cost vector changes.
            # DPP keeps the canonicalization cached, so this forwards just a new
            # objective to MOSEK rather than rebuilding the model.
            self.objective_coeffs.value = self.objective_vectors_by_y[y]
            value = solve_cvxpy_problem_preserving_duals(problem, solve_kwargs)
            assert_cvxpy_solution_is_optimal(problem=problem, value=value, problem_kind=f"LP for y={y}")
            out[y] = float(value)
            self.cvxpy_problems_by_y[y] = problem
            self.dual_values_by_y[y] = snapshot_dual_values(self.dual_constraints)

        self.eve_guess_by_y = out
        self.solution_probabilities = self._extract_solution_probabilities()
        return out.copy()

    def solve(
        self,
        where_key: Sequence[Sequence[int]] | None = None,
        *,
        master_key_holder: Literal["Alice", "Bob"] | str | None = None,
        solver: str | None = None,
    ) -> np.ndarray:
        return self.solve_lp(where_key=where_key, master_key_holder=master_key_holder, solver=solver)

    def _build_problem(self) -> None:
        self.cvxpy_variable = cp.Variable(self.num_variables, name="P")
        self.cvxpy_constraints = []
        self.cvxpy_problem = None
        self.cvxpy_problems_by_y = {}
        self.dual_constraints = {}
        self.dual_values_by_y = {}
        self.objective_vectors_by_y = {}
        self.solution_probabilities = None

        # Nonnegativity stays a single named vector inequality. Its dual is the
        # reduced-cost vector, which we expose intact under one key.
        self._add_constraint(("probability_nonnegative",), self.cvxpy_variable >= 0.0)

        # Equalities become a handful of stacked `A @ P == b` blocks instead of
        # thousands of scalar constraints (see module docstring).
        self._add_observed_bob_block()
        self._add_operational_equivalence_block()
        self._add_invalid_guess_block()

        # Parameterized objective: build the model ONCE, then re-solve per y by
        # only assigning `objective_coeffs.value`. `coeffs @ P` is DPP-compliant
        # (a parameter-affine objective), so CVXPY caches the canonicalization
        # across solves and only ships a fresh cost vector to MOSEK.
        self.objective_coeffs = cp.Parameter(self.num_variables, name="objective_coeffs")
        self.cvxpy_problem = cp.Problem(
            cp.Maximize(self.objective_coeffs @ self.cvxpy_variable),
            self.cvxpy_constraints,
        )
        self._build_objective_vectors()

    def _add_observed_bob_block(self) -> None:
        """Data consistency: sum_e P(b, e | x, y) = p_data(b | x, y)."""
        data = np.asarray(self.scenario.data_numeric, dtype=float)
        num_rows = self.num_x * self.num_y * self.num_b
        # Row r enumerates (x, y, b) in C-order; the num_e columns it sums over
        # are exactly the contiguous block [r * num_e : (r + 1) * num_e].
        rows = np.repeat(np.arange(num_rows), self.num_e)
        cols = np.arange(num_rows * self.num_e)
        vals = np.ones(num_rows * self.num_e)
        rhs = data.reshape(num_rows)
        xs, ys, bs = np.unravel_index(np.arange(num_rows), (self.num_x, self.num_y, self.num_b))
        keys = [("observed_bob", int(x), int(y), int(b)) for x, y, b in zip(xs, ys, bs)]
        self._add_matrix_block("observed_bob", rows, cols, vals, rhs, keys)

    def _add_operational_equivalence_block(self) -> None:
        """All preparation/measurement OPEQ rows stacked into one `A @ P == 0`."""
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        keys: list[DualKey] = []
        offset = 0

        # Preparation OPEQs: sum_x c[x] P(b, e | x, y) = 0 for every (y, b, e).
        prep_rows = self.num_y * self.num_b * self.num_e
        py, pb, pe = np.unravel_index(np.arange(prep_rows), (self.num_y, self.num_b, self.num_e))
        for opeq_index, coeffs_raw in enumerate(self.scenario.opeq_preps_numeric):
            coeffs = np.asarray(coeffs_raw, dtype=float)
            x_nonzero = np.flatnonzero(coeffs)
            if x_nonzero.size == 0:
                continue
            for x in x_nonzero.tolist():
                rows.append(np.arange(prep_rows) + offset)
                cols.append(self._flat_index(x, py, pb, pe))
                vals.append(np.full(prep_rows, float(coeffs[x])))
            keys.extend(
                ("prep_opeq", opeq_index, int(y), int(b), int(e)) for y, b, e in zip(py, pb, pe)
            )
            offset += prep_rows

        # Measurement OPEQs: sum_{y,b} d[y,b] P(b, e | x, y) = 0 for every (x, e).
        meas_rows = self.num_x * self.num_e
        mx, me = np.unravel_index(np.arange(meas_rows), (self.num_x, self.num_e))
        for opeq_index, coeffs_raw in enumerate(self.scenario.opeq_meas_numeric):
            coeffs = np.asarray(coeffs_raw, dtype=float)
            y_nonzero, b_nonzero = np.nonzero(coeffs)
            if y_nonzero.size == 0:
                continue
            for y, b in zip(y_nonzero.tolist(), b_nonzero.tolist()):
                rows.append(np.arange(meas_rows) + offset)
                cols.append(self._flat_index(mx, y, b, me))
                vals.append(np.full(meas_rows, float(coeffs[y, b])))
            keys.extend(("measurement_opeq", opeq_index, int(x), int(e)) for x, e in zip(mx, me))
            offset += meas_rows

        if offset == 0:
            return
        self._add_matrix_block(
            "operational_equivalences",
            np.concatenate(rows),
            np.concatenate(cols),
            np.concatenate(vals),
            np.zeros(offset),
            keys,
        )

    def _add_invalid_guess_block(self) -> None:
        """Force invalid guess labels to carry zero mass when B(y) < B_max."""
        cols: list[int] = []
        keys: list[DualKey] = []
        for x, y in np.ndindex(self.num_x, self.num_y):
            b_count = int(self.b_cardinality_per_y[y])
            for b in range(b_count):
                for e in range(b_count, self.num_e):
                    cols.append(int(self._flat_index(x, y, b, e)))
                    keys.append(("invalid_guess", int(x), int(y), int(b), int(e)))
        if not cols:
            return
        num_rows = len(cols)
        self._add_matrix_block(
            "invalid_guess",
            np.arange(num_rows),
            np.asarray(cols, dtype=int),
            np.ones(num_rows),
            np.zeros(num_rows),
            keys,
        )

    def _build_objective_vectors(self) -> None:
        """Precompute the per-y cost vector assigned to ``objective_coeffs``."""
        key_lookup = self.scenario.key_selection_by_xy
        for y in range(self.num_y):
            vec = np.zeros(self.num_variables, dtype=float)
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
                    vec[int(self._flat_index(int(x), y, b, e))] += weight
            self.objective_vectors_by_y[y] = vec

    def _add_matrix_block(
        self,
        name: str,
        rows: np.ndarray,
        cols: np.ndarray,
        vals: np.ndarray,
        rhs: np.ndarray,
        keys: list[DualKey],
    ) -> None:
        """Register one stacked equality `A @ P == rhs` from sparse triplets."""
        matrix = sp.csr_matrix(
            (np.asarray(vals, dtype=float), (np.asarray(rows), np.asarray(cols))),
            shape=(len(rhs), self.num_variables),
        )
        constraint = matrix @ self.cvxpy_variable == np.asarray(rhs, dtype=float)
        register_matrix_constraint(
            name=name,
            constraint=constraint,
            row_keys=keys,
            constraints=self.cvxpy_constraints,
            dual_constraints=self.dual_constraints,
        )

    def _flat_index(self, x, y, b, e):
        """Vectorized ravel of (x, y, b, e) into the flat variable index."""
        return ((x * self.num_y + y) * self.num_b + b) * self.num_e + e

    def _where_key_row(self, y: int) -> np.ndarray:
        return np.asarray(self.where_key[int(y)], dtype=int).reshape(-1)

    def _add_constraint(self, key: DualKey, constraint: cp.Constraint) -> None:
        add_named_constraint(
            key=key,
            constraint=constraint,
            constraints=self.cvxpy_constraints,
            dual_constraints=self.dual_constraints,
        )

    def _solve_kwargs(self) -> dict[str, object]:
        # `warm_start` is harmless but secondary: the real reuse comes from the
        # DPP parameter cache above. We deliberately leave MOSEK's presolve
        # (linear-dependency detection, variable elimination) at its defaults.
        solve_kwargs: dict[str, object] = {
            "solver": self.solver,
            "verbose": self.verbose >= 2,
            "warm_start": True,
        }
        mosek_params: dict[str, object] = {}
        if is_mosek_solver(self.solver) and self.threads is not None and self.threads > 0:
            mosek_params["MSK_IPAR_NUM_THREADS"] = int(self.threads)
        if mosek_params:
            solve_kwargs["mosek_params"] = mosek_params
        return solve_kwargs

    def _extract_solution_probabilities(self) -> np.ndarray | None:
        if self.cvxpy_variable is None or self.cvxpy_variable.value is None:
            return None
        return np.asarray(self.cvxpy_variable.value, dtype=float).reshape(
            (self.num_x, self.num_y, self.num_b, self.num_e)
        )

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

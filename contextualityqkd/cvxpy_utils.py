"""Shared CVXPY helpers for Eve optimization backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Hashable

import cvxpy as cp
import numpy as np


DualKey = tuple[Hashable, ...]


def add_named_constraint(
    *,
    key: DualKey,
    constraint: cp.Constraint,
    constraints: list[cp.Constraint],
    dual_constraints: dict[DualKey, cp.Constraint],
) -> None:
    """Append a CVXPY constraint and keep a stable handle for dual inspection.

    Used by the SDP backend, where each semantic key maps to one constraint and
    the dual is exposed intact (see :class:`NamedDuals` for the array-valued
    style the LP backend uses).
    """
    if key in dual_constraints:
        raise RuntimeError(f"Duplicate CVXPY dual constraint key: {key!r}")
    constraints.append(constraint)
    dual_constraints[key] = constraint


def snapshot_dual_values(
    dual_constraints: dict[DualKey, cp.Constraint],
) -> dict[DualKey, object | None]:
    """Capture currently available CVXPY dual values by semantic constraint key."""
    return {key: constraint.dual_value for key, constraint in dual_constraints.items()}


class NamedDuals:
    """Collect CVXPY constraints under physical names and report array duals.

    Each named group's dual is returned as a single numpy array whose axes are
    that constraint's natural physical indices, so callers slice
    ``duals["observed_bob"][x, y, b]`` instead of looking up a flat tuple key.
    CVXPY already returns ``constraint.dual_value`` shaped like the constraint,
    so there is nothing to map.

    Three registration styles:

    * :meth:`add` -- one constraint reported as its own shaped dual array.
    * :meth:`add_group` -- a list of like-shaped constraints whose duals are
      stacked along a new leading axis (e.g. one row per operational equivalence).
    * :meth:`add_masked` -- a constraint over masked cells whose 1-D dual is
      scattered back into the full ``mask.shape`` array, ``NaN`` off the mask so
      "never constrained" is distinguishable from a genuine zero multiplier.
    """

    def __init__(self) -> None:
        self._single: dict[str, cp.Constraint] = {}
        self._groups: dict[str, list[cp.Constraint]] = {}
        self._masked: dict[str, tuple[cp.Constraint, np.ndarray]] = {}

    def add(self, name: str, constraint: cp.Constraint) -> cp.Constraint:
        self._claim(name)
        self._single[name] = constraint
        return constraint

    def add_group(self, name: str, group: Sequence[cp.Constraint]) -> None:
        self._claim(name)
        self._groups[name] = list(group)

    def add_masked(self, name: str, constraint: cp.Constraint, mask: np.ndarray) -> cp.Constraint:
        self._claim(name)
        self._masked[name] = (constraint, np.asarray(mask, dtype=bool))
        return constraint

    @property
    def constraints(self) -> list[cp.Constraint]:
        """Every registered constraint, ready to hand to ``cp.Problem``."""
        return (
            list(self._single.values())
            + [c for group in self._groups.values() for c in group]
            + [c for c, _ in self._masked.values()]
        )

    def snapshot(self) -> dict[str, np.ndarray | None]:
        """Capture the current dual of every group as a physically-indexed array."""
        out: dict[str, np.ndarray | None] = {}
        for name, constraint in self._single.items():
            dual = constraint.dual_value
            out[name] = None if dual is None else np.asarray(dual, dtype=float)
        for name, group in self._groups.items():
            duals = [c.dual_value for c in group]
            if not group or any(d is None for d in duals):
                out[name] = None
            else:
                out[name] = np.stack([np.asarray(d, dtype=float) for d in duals], axis=0)
        for name, (constraint, mask) in self._masked.items():
            dual = constraint.dual_value
            if dual is None:
                out[name] = None
            else:
                full = np.full(mask.shape, np.nan, dtype=float)
                full[mask] = np.asarray(dual, dtype=float).reshape(-1)
                out[name] = full
        return out

    def _claim(self, name: str) -> None:
        if name in self._single or name in self._groups or name in self._masked:
            raise RuntimeError(f"Duplicate dual group name: {name!r}")


def assert_cvxpy_solution_is_optimal(
    *,
    problem: cp.Problem,
    value: float | None,
    problem_kind: str,
) -> None:
    """Raise unless CVXPY returned an optimal finite objective value."""
    if problem.status == cp.OPTIMAL and value is not None and np.isfinite(float(value)):
        return
    raise RuntimeError(
        f"{problem_kind} solve failed: CVXPY did not return an optimal finite solution. "
        f"status={problem.status!r}, value={value!r}, solver_stats={problem.solver_stats!r}"
    )


def is_mosek_solver(solver: object) -> bool:
    """Return whether a CVXPY solver token denotes MOSEK."""
    return str(solver).upper() == str(cp.MOSEK).upper()


_BACKEND_SOLVERS = {
    "mosek": cp.MOSEK,          # interior-point: dense but symmetric dual
    "mosek_simplex": cp.MOSEK,  # dual simplex (+ bfs): sparse vertex dual
    "highs": cp.HIGHS,          # simplex: sparse vertex dual
    "simplex": cp.HIGHS,
    "clarabel": cp.CLARABEL,
    "scipy": cp.SCIPY,
    "scs": cp.SCS,
}


def resolve_backend_solver(backend_solver: str) -> str:
    """Map a friendly backend name to a CVXPY solver token."""
    token = str(backend_solver).strip().lower()
    if token not in _BACKEND_SOLVERS:
        raise ValueError(
            f"backend_solver must be one of {sorted(_BACKEND_SOLVERS)}, got {backend_solver!r}."
        )
    return _BACKEND_SOLVERS[token]


def uses_mosek_simplex(backend_solver: str) -> bool:
    """Whether a backend name requests MOSEK's (dual) simplex optimizer."""
    return str(backend_solver).strip().lower() == "mosek_simplex"


def build_solve_kwargs(
    solver: str,
    *,
    mosek_simplex: bool = False,
    threads: int | None = None,
    verbose: bool = False,
) -> dict[str, object]:
    """Assemble ``problem.solve`` kwargs shared by the LP backends.

    For MOSEK simplex we set the dual-simplex optimizer and ``bfs=True`` so
    CVXPY reads MOSEK's basic (vertex) solution, which yields a sparse dual
    witness.
    """
    # These backends model with N-dimensional (>2-D) array variables, which the
    # C++ canonicalization backend does not support; choose the SCIPY backend
    # explicitly so CVXPY does not emit a "defaulting to SCIPY backend" warning.
    solve_kwargs: dict[str, object] = {
        "solver": solver,
        "verbose": bool(verbose),
        "canon_backend": cp.SCIPY_CANON_BACKEND,
    }
    mosek_params: dict[str, object] = {}
    if is_mosek_solver(solver):
        if mosek_simplex:
            mosek_params["MSK_IPAR_OPTIMIZER"] = "MSK_OPTIMIZER_DUAL_SIMPLEX"
            solve_kwargs["bfs"] = True
        if threads is not None and threads > 0:
            mosek_params["MSK_IPAR_NUM_THREADS"] = int(threads)
    if mosek_params:
        solve_kwargs["mosek_params"] = mosek_params
    return solve_kwargs


def _default_pbxy_label(index: tuple[int, ...]) -> str:
    """Default term label for a ``(x, y, b)`` index into ``P(b|x,y)``."""
    x, y, b = (int(i) for i in index)
    return f"P({b}|{x},{y})"


def coefficient_value_groups(
    coeffs: np.ndarray,
    *,
    atol: float = 1e-9,
    decimals: int = 6,
) -> list[tuple[float, list[tuple[int, ...]]]]:
    """Group nonzero coefficients by (rounded) value.

    ``coeffs`` is an array shaped like ``P(b|x,y)``. Returns a list of
    ``(value, index_tuples)`` pairs, sorted by descending value, where each
    ``index_tuple`` is a multi-index of ``coeffs`` carrying that value. This is
    the shared core for turning an LP dual into a readable inequality.
    """
    arr = np.asarray(coeffs, dtype=float)
    flat = arr.reshape(-1)
    nonzero = np.flatnonzero(np.abs(flat) > atol)
    if nonzero.size == 0:
        return []
    rounded = np.round(flat[nonzero], decimals)
    groups: list[tuple[float, list[tuple[int, ...]]]] = []
    for value in sorted(set(rounded.tolist()), reverse=True):
        selected = nonzero[rounded == value]
        coords = [tuple(int(c) for c in np.unravel_index(i, arr.shape)) for i in selected]
        groups.append((float(value), coords))
    return groups


def format_coefficient_groups(
    coeffs: np.ndarray,
    *,
    coord_label=_default_pbxy_label,
    precision: int = 4,
    atol: float = 1e-9,
    indent: str = "    ",
) -> str:
    """Render an inequality's coefficients grouped by distinct value.

    One block per distinct coefficient value (descending), listing the
    ``P(b|x,y)`` terms that share it -- far more readable than a dense array
    for sparse/degenerate witnesses.
    """
    groups = coefficient_value_groups(coeffs, atol=atol, decimals=precision)
    if not groups:
        return f"{indent}(all coefficients zero)"
    lines = []
    for value, coords in groups:
        terms = ", ".join(coord_label(c) for c in coords)
        lines.append(f"{indent}{format(value, f'+.{precision}g')} * [ {terms} ]")
    return "\n".join(lines)


def solve_cvxpy_problem_preserving_duals(
    problem: cp.Problem,
    solve_kwargs: dict[str, object],
) -> float:
    """Solve through CVXPY while preserving available duals.

    CVXPY 1.9 can solve some MOSEK cone models but raise a ``KeyError`` while
    reshaping missing dual entries. This local patch keeps all returned dual
    values and skips only IDs that the solver adapter did not provide.
    """

    from cvxpy import settings as s
    from cvxpy.constraints import ExpCone, PSD, SOC
    from cvxpy.reductions.complex2real.complex2real import Complex2Real
    from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
    from cvxpy.reductions.solution import Solution

    original_complex_invert = Complex2Real.invert
    original_invert = ConeMatrixStuffing.invert

    complex_globals = original_complex_invert.__globals__
    equality_cls = complex_globals["Equality"]
    zero_cls = complex_globals["Zero"]
    vec_to_upper_tri = complex_globals["vec_to_upper_tri"]

    def invert_skipping_missing_duals(
        self: ConeMatrixStuffing,
        solution: object,
        inverse_data: object,
    ) -> object:
        var_map = inverse_data.var_offsets
        con_map = inverse_data.cons_id_map
        opt_val = solution.opt_val
        if solution.status not in s.ERROR and not inverse_data.minimize:
            opt_val = -solution.opt_val

        dual_vars: dict[object, object] = {}
        if solution.dual_vars is not None:
            for old_con, new_con in con_map.items():
                con_obj = inverse_data.id2cons[old_con]
                shape = con_obj.shape
                dual_value = solution.dual_vars.get(new_con)
                if dual_value is None:
                    continue
                if shape == () or isinstance(con_obj, (ExpCone, SOC)):
                    dual_vars[old_con] = dual_value
                else:
                    dual_vars[old_con] = np.reshape(dual_value, shape, order="F")

        primal_vars: dict[object, object] = {}
        if solution.status not in s.SOLUTION_PRESENT:
            return Solution(solution.status, opt_val, primal_vars, dual_vars, solution.attr)

        x_opt = list(solution.primal_vars.values())[0]
        for var_id, offset in var_map.items():
            shape = inverse_data.var_shapes[var_id]
            size = np.prod(shape, dtype=int)
            primal_vars[var_id] = np.reshape(x_opt[offset : offset + size], shape, order="F")

        if solution.dual_vars is not None:
            for old_con, new_con in con_map.items():
                if new_con not in solution.dual_vars:
                    continue
                con_obj = inverse_data.id2cons[old_con]
                shape = con_obj.shape
                if shape == () or isinstance(con_obj, (ExpCone, SOC)) or (
                    isinstance(con_obj, PSD) and con_obj.num_cones() > 1
                ):
                    dual_vars[old_con] = solution.dual_vars[new_con]
                else:
                    dual_vars[old_con] = np.reshape(solution.dual_vars[new_con], shape, order="F")

        return Solution(solution.status, opt_val, primal_vars, dual_vars, solution.attr)

    def complex_invert_skipping_missing_duals(
        self: Complex2Real,
        solution: object,
        inverse_data: object,
    ) -> object:
        pvars: dict[object, object] = {}
        dvars: dict[object, object] = {}
        if solution.status in s.SOLUTION_PRESENT:
            for vid, var in inverse_data.id2var.items():
                if var.is_real():
                    pvars[vid] = solution.primal_vars[vid]
                elif var in self.canon_methods._variables:
                    real_var, imag_var = self.canon_methods._variables[var]
                    if var.is_imag():
                        pvars[vid] = 1j * solution.primal_vars[imag_var.id]
                    elif var.is_hermitian():
                        pvars[vid] = solution.primal_vars[real_var.id]
                        if imag_var is not None and imag_var.id in solution.primal_vars:
                            imag_val = solution.primal_vars[imag_var.id]
                            imag_val = vec_to_upper_tri(imag_val, True).value
                            imag_val -= imag_val.T
                            pvars[vid] = pvars[vid] + 1j * imag_val
                    else:
                        pvars[vid] = solution.primal_vars[real_var.id]
                        if imag_var.id in solution.primal_vars:
                            pvars[vid] = pvars[vid] + 1j * solution.primal_vars[imag_var.id]

            if solution.dual_vars:
                for cid, cons in inverse_data.id2cons.items():
                    if cons.is_real():
                        if cid in solution.dual_vars:
                            dvars[cid] = solution.dual_vars[cid]
                    elif cons.is_imag():
                        imag_id = inverse_data.real2imag[cid]
                        if imag_id in solution.dual_vars:
                            dvars[cid] = 1j * solution.dual_vars[imag_id]
                    elif isinstance(cons, (equality_cls, zero_cls)):
                        imag_id = inverse_data.real2imag[cid]
                        if cid not in solution.dual_vars:
                            continue
                        if imag_id in solution.dual_vars:
                            dvars[cid] = solution.dual_vars[cid] + 1j * solution.dual_vars[imag_id]
                        else:
                            dvars[cid] = solution.dual_vars[cid]
                    elif isinstance(cons, PSD):
                        if cid not in solution.dual_vars:
                            continue
                        n = cons.args[0].shape[0]
                        dual = solution.dual_vars[cid]
                        dvars[cid] = dual[:n, :n] + 1j * dual[n:, :n]
                    elif isinstance(cons, self.UNIMPLEMENTED_COMPLEX_DUALS):
                        pass
                    else:
                        raise Exception("Unknown constraint type.")

        return Solution(solution.status, solution.opt_val, pvars, dvars, solution.attr)

    Complex2Real.invert = complex_invert_skipping_missing_duals
    ConeMatrixStuffing.invert = invert_skipping_missing_duals
    try:
        return float(problem.solve(**solve_kwargs))
    finally:
        ConeMatrixStuffing.invert = original_invert
        Complex2Real.invert = original_complex_invert

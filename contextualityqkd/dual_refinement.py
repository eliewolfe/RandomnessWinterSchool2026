"""L2-norm refinement of LP dual certificates.

Each LP in the package has a dual optimal face that is typically a flat polytope
(many vertices map to the same primal optimum). Vertex simplex solvers pick an
arbitrary corner; the resulting witness inequality is not the most readable one.
This module projects the raw LP dual onto the optimal dual face by solving a
secondary QP that minimizes the squared ell-2 norm of only the *witness-relevant*
duals, subject to dual feasibility and strong duality. The minimum-norm point of
a convex set is unique, so symmetry-induced equalities surface automatically.

For each of the three LPs (contextual fraction, dephasing robustness, Eve
guessing per y) the witness duals enter the QP objective; all auxiliary duals
(OPEQ groups, invalid-guess mask, lambda<=1 multiplier) appear only in the
dual-feasibility constraints. The QP is solved with a configurable solver and
an internal fallback chain.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Iterable

import cvxpy as cp
import numpy as np


# --------------------------------------------------------------------------- #
# QP solver resolution + fallback chain                                        #
# --------------------------------------------------------------------------- #

_QP_SOLVERS: dict[str, str] = {
    "gurobi": cp.GUROBI,
    "gurobi_simplex": cp.GUROBI,    # _simplex is LP-only; degrades to barrier QP
    "mosek": cp.MOSEK,
    "mosek_simplex": cp.MOSEK,      # _simplex is LP-only; degrades to barrier QP
    "clarabel": cp.CLARABEL,
    "osqp": cp.OSQP,
    "scs": cp.SCS,
}

# LP-only friendly names that cannot solve a strictly convex QP; we warn and
# fall through to the default chain instead.
_LP_ONLY_NAMES = frozenset({"highs", "simplex", "scipy"})

# Order after the user-requested solver, used both to dedupe and to fill out
# the chain. Gurobi leads because it is machine-precision tight on strong
# duality (~3e-16) and reproduces the same symmetric coefficient grouping as
# OSQP at the print precision (4 decimals) commonly used downstream. OSQP
# remains a strong alternative — at print precision >= 6 it produces exact
# rational symmetries (e.g. 1/8, 7/72, 1/24, 1/72 on the hexagon Eve LP)
# where Gurobi/MOSEK exhibit ~1e-6 floating-point noise — and is selectable
# via qp_solver="osqp".
_QP_DEFAULT_CHAIN: tuple[str, ...] = (cp.GUROBI, cp.MOSEK, cp.OSQP, cp.CLARABEL, cp.SCS)


def resolve_qp_solver_chain(requested: str) -> list[str]:
    """Return the ordered list of CVXPY solver tokens to try for the QP.

    ``requested`` is a friendly backend name from ``_QP_SOLVERS`` or one of the
    LP-only aliases ``highs``/``simplex``/``scipy`` (which is dropped with a
    warning). The chain always falls through to GUROBI -> MOSEK -> CLARABEL ->
    OSQP -> SCS after the requested entry, deduped against the requested token.
    """
    token = str(requested).strip().lower()
    chain: list[str] = []
    if token in _QP_SOLVERS:
        chain.append(_QP_SOLVERS[token])
    elif token in _LP_ONLY_NAMES:
        warnings.warn(
            f"qp_solver={requested!r} is LP-only and cannot solve the dual-refinement QP; "
            "falling through to the default QP chain.",
            RuntimeWarning,
            stacklevel=3,
        )
    else:
        raise ValueError(
            f"qp_solver must be one of {sorted(_QP_SOLVERS) + sorted(_LP_ONLY_NAMES)}, "
            f"got {requested!r}."
        )
    for default in _QP_DEFAULT_CHAIN:
        if default not in chain:
            chain.append(default)
    # Drop any token whose backend is not installed in this environment.
    installed = set(cp.installed_solvers())
    return [tok for tok in chain if tok in installed]


# --------------------------------------------------------------------------- #
# QP solve loop with fallback                                                  #
# --------------------------------------------------------------------------- #


def _solve_qp_with_fallback(
    problem: cp.Problem,
    *,
    qp_solver: str,
    primal_label: str,
    threads: int | None = None,
    verbose: bool = False,
) -> float:
    """Solve a prebuilt QP, walking the configured fallback chain on failure."""
    chain = resolve_qp_solver_chain(qp_solver)
    if not chain:
        raise RuntimeError(
            f"No installed QP solver available for {primal_label}; install one of "
            f"{[tok for tok in _QP_DEFAULT_CHAIN]}."
        )
    last_status: str | None = None
    last_err: Exception | None = None
    for token in chain:
        # 4-D dual variables in the Eve QP need the SCIPY canon backend; passing
        # it for all QPs is harmless and silences the "defaulting to SCIPY"
        # warning that fires when a problem has a >2-D expression.
        kwargs: dict[str, object] = {
            "solver": token,
            "verbose": bool(verbose),
            "canon_backend": cp.SCIPY_CANON_BACKEND,
        }
        if token == cp.GUROBI and threads is not None and threads > 0:
            kwargs["Threads"] = int(threads)
        if token == cp.MOSEK and threads is not None and threads > 0:
            kwargs["mosek_params"] = {"MSK_IPAR_NUM_THREADS": int(threads)}
        if token == cp.OSQP:
            # CVXPY's OSQP defaults (eps=1e-5, max_iter=10000) are too loose
            # for strong-duality identities to hold to ~6 decimal places on
            # dense scenarios (Cabello CF saw a 1e-3 gap at defaults). 1e-8
            # gets strong duality to ~1e-7 in practice while staying fast on
            # the per-y Eve QP. The symmetric-rational structure OSQP recovers
            # on the Eve hexagon is unaffected by tightening eps further.
            kwargs.update(eps_abs=1e-8, eps_rel=1e-8, max_iter=20000)
        try:
            value = float(problem.solve(**kwargs))
        except (cp.error.SolverError, KeyError, ValueError) as exc:
            last_err = exc
            continue
        last_status = problem.status
        if problem.status == cp.OPTIMAL:
            return value
        if problem.status == cp.OPTIMAL_INACCURATE:
            warnings.warn(
                f"QP refinement for {primal_label} returned OPTIMAL_INACCURATE on solver "
                f"{token!r}; accepting the refined dual but expect ~solver-tolerance error.",
                RuntimeWarning,
                stacklevel=3,
            )
            return value
    err_suffix = f"; last error: {last_err!r}" if last_err is not None else ""
    raise RuntimeError(
        f"Dual-refinement QP for {primal_label} failed across all backends "
        f"({chain}). Last status: {last_status!r}{err_suffix}."
    )


def refine_witness_duals(
    *,
    dual_constraints: list[cp.Constraint],
    witness_duals: Sequence[cp.Variable],
    qp_solver: str,
    primal_label: str,
    threads: int | None = None,
    verbose: bool = False,
) -> tuple[float, list[np.ndarray]]:
    """Build and solve  min sum_k ||witness_duals[k]||^2  s.t. dual_constraints.

    ``dual_constraints`` must already encode dual feasibility plus the
    strong-duality equality that pins the dual face to the primal optimum.
    Auxiliary dual variables (those not in ``witness_duals``) live as ordinary
    ``cp.Variable``s inside the constraint list.
    """
    objective = cp.Minimize(sum((cp.sum_squares(v) for v in witness_duals), start=cp.Constant(0.0)))
    problem = cp.Problem(objective, dual_constraints)
    value = _solve_qp_with_fallback(
        problem,
        qp_solver=qp_solver,
        primal_label=primal_label,
        threads=threads,
        verbose=verbose,
    )
    return value, [np.asarray(v.value, dtype=float) for v in witness_duals]


# --------------------------------------------------------------------------- #
# Per-LP dual program builders                                                 #
# --------------------------------------------------------------------------- #


def _prep_effect_reduced_cost_matrix(
    *,
    coeff_xyb: cp.Expression,
    prep_extremals: np.ndarray,
    effect_extremals: np.ndarray,
) -> cp.Expression:
    """Reduced cost on ``w[i,j]`` from a per-(x,y,b) dual coefficient.

    The primal subbehavior is ``S(x,y,b) = sum_ij w_ij prep_i(x) effect_j(y,b)``;
    a Lagrangian term of the form ``-sum_xyb coeff[x,y,b] S[x,y,b]`` contributes
    ``-sum_xyb coeff[x,y,b] prep_i(x) effect_j(y,b)`` to the coefficient of
    ``w[i,j]`` in ``L``. We return that contraction with the leading sign so the
    caller can read the dual-feasibility inequality directly.

    Returned shape ``(N_prep, N_eff)``.
    """
    n_prep, num_x = prep_extremals.shape
    n_eff = effect_extremals.shape[0]
    num_y, num_b = effect_extremals.shape[1:]
    # coeff_xyb is shape (X, Y, B); reduce to (X, N_eff) by contracting (Y, B)
    # against each effect ray.
    effect_flat = effect_extremals.reshape(n_eff, -1)            # (N_eff, Y*B)
    coeff_flat = cp.reshape(coeff_xyb, (num_x, num_y * num_b), order="C")
    # coeff_x_eff[x, j] = sum_yb coeff[x,y,b] effect[j,y,b] = (coeff_flat @ effect_flat.T)
    coeff_x_eff = coeff_flat @ effect_flat.T                     # (X, N_eff)
    # Contract x against prep_extremals[i, x] -> shape (N_prep, N_eff)
    return prep_extremals @ coeff_x_eff


def build_dual_contextual_fraction(
    *,
    data: np.ndarray,
    prep_extremals: np.ndarray,
    effect_extremals: np.ndarray,
    primal_value: float,
    qp_solver: str,
    threads: int | None = None,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """Refine the contextual-fraction LP duals to the min-norm point of the face.

    Primal: max lambda  s.t.  S(w) <= data  (dual mu>=0),
                              sum_b S(w) == lambda  (dual nu free),
                              lambda <= 1  (dual kappa>=0),
                              w >= 0.

    Witness functional alpha = -(mu + nu[:,:,None]); strong duality says
    sum mu*data + kappa = lambda* = primal_value.

    Returns a dict with refined ``subbehavior_le_data`` (mu) and ``uniform_mass``
    (nu) arrays whose shapes mirror the LP-side dual snapshot.
    """
    num_x, num_y, num_b = data.shape
    mu = cp.Variable((num_x, num_y, num_b), nonneg=True, name="mu_refined")
    nu = cp.Variable((num_x, num_y), name="nu_refined")
    kappa = cp.Variable(nonneg=True, name="kappa_refined")

    constraints: list[cp.Constraint] = []

    # Stationarity w.r.t. lambda (free except for kappa>=0 enforcing lambda<=1):
    constraints.append(1.0 + cp.sum(nu) - kappa == 0.0)

    # Dual feasibility on each w[i,j] (w>=0, primal cost 0, sense max):
    # sum_xyb mu[x,y,b] prep[i,x] effect[j,y,b]
    #   + sum_xy  nu[x,y] prep[i,x] sum_b effect[j,y,b]   >=  0
    if prep_extremals.size and effect_extremals.size:
        # Term from mu: (N_prep, N_eff) via the shared helper.
        mu_rc = _prep_effect_reduced_cost_matrix(
            coeff_xyb=mu, prep_extremals=prep_extremals, effect_extremals=effect_extremals,
        )
        # Term from nu: nu has shape (X, Y); contract Y against effect.sum(b) of
        # shape (N_eff, Y), then X against prep.
        effect_sum_b = effect_extremals.sum(axis=2)              # (N_eff, Y)
        nu_x_eff = nu @ effect_sum_b.T                           # (X, N_eff)
        nu_rc = prep_extremals @ nu_x_eff                        # (N_prep, N_eff)
        constraints.append(mu_rc + nu_rc >= 0.0)

    # Strong duality: <mu, data> + kappa == lambda*.
    constraints.append(cp.sum(cp.multiply(mu, data)) + kappa == float(primal_value))

    _value, (mu_val, nu_val) = refine_witness_duals(
        dual_constraints=constraints,
        witness_duals=(mu, nu),
        qp_solver=qp_solver,
        primal_label="contextual_fraction",
        threads=threads,
        verbose=verbose,
    )
    return {
        "subbehavior_le_data": np.asarray(mu_val, dtype=float),
        "uniform_mass": np.asarray(nu_val, dtype=float),
    }


def build_dual_dephasing_robustness(
    *,
    data: np.ndarray,
    target: np.ndarray,
    prep_extremals: np.ndarray,
    effect_extremals: np.ndarray,
    primal_value: float,
    qp_solver: str,
    threads: int | None = None,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """Refine the dephasing-robustness LP equality dual to min-norm.

    Primal: min r  s.t.  S(w) == (1-r) data + r target  (dual gamma free,
                                                         CVXPY sign convention),
                          w >= 0.

    Existing convention (matches ``contextuality.py:261``): the printed witness
    is alpha = -gamma. Strong duality in the CVXPY sign reads
    ``<gamma, data> == -r*`` and dual feasibility on r free is
    ``<gamma, data - target> == -1``. The QP enforces both as equalities plus
    the per-(i,j) reduced-cost inequality on w.

    Returns ``{"dephased_behavior": gamma_refined}`` so downstream code that
    flips the sign reads the right witness.
    """
    if data.shape != target.shape:
        raise ValueError(
            f"dephasing target shape {target.shape} must match data shape {data.shape}."
        )
    gamma = cp.Variable(data.shape, name="gamma_refined")

    constraints: list[cp.Constraint] = []
    # Stationarity w.r.t. r (free):
    constraints.append(cp.sum(cp.multiply(gamma, data - target)) == -1.0)

    # Dual feasibility on w[i,j] (sense min, w>=0): primal cost 0, so reduced
    # cost from -gamma*S in L: coefficient of w[i,j] is
    # sum_xyb gamma[x,y,b] prep[i,x] effect[j,y,b], and a min-LP with w>=0
    # requires this coefficient >= 0.
    if prep_extremals.size and effect_extremals.size:
        gamma_rc = _prep_effect_reduced_cost_matrix(
            coeff_xyb=gamma, prep_extremals=prep_extremals, effect_extremals=effect_extremals,
        )
        constraints.append(gamma_rc >= 0.0)

    # Strong duality: <gamma, data> == -r* = -primal_value.
    constraints.append(cp.sum(cp.multiply(gamma, data)) == -float(primal_value))

    _value, (gamma_val,) = refine_witness_duals(
        dual_constraints=constraints,
        witness_duals=(gamma,),
        qp_solver=qp_solver,
        primal_label="dephasing_robustness",
        threads=threads,
        verbose=verbose,
    )
    return {"dephased_behavior": np.asarray(gamma_val, dtype=float)}


# --------------------------------------------------------------------------- #
# Eve LP per-y refiner (DPP-parameterized)                                     #
# --------------------------------------------------------------------------- #


class EveDualRefiner:
    """Parameterized QP that refines Eve's per-y guessing witness c_y.

    The Eve LP solves ``num_y`` separate primal problems sharing the same
    feasible region (only the linear objective changes). The refinement QP
    inherits the same structure: per-y we vary ``obj_coeffs`` and ``G(y)``
    only. Both enter as ``cp.Parameter``s, so the QP canonicalizes once and
    every per-y solve reuses the cached factorization.
    """

    def __init__(
        self,
        *,
        data: np.ndarray,
        prep_opeq: np.ndarray,
        meas_opeq: np.ndarray,
        invalid_guess_mask: np.ndarray,
        qp_solver: str,
        threads: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.data = np.asarray(data, dtype=float)
        self.prep_opeq = np.asarray(prep_opeq, dtype=float)
        self.meas_opeq = np.asarray(meas_opeq, dtype=float)
        self.invalid_guess_mask = np.asarray(invalid_guess_mask, dtype=bool)
        self.qp_solver = str(qp_solver).strip().lower()
        self.threads = None if threads is None else int(threads)
        self.verbose = bool(verbose)

        num_x, num_y, num_b = self.data.shape
        # Eve guesses share Bob's outcome alphabet size at this LP level
        num_e = num_b
        self.shape = (num_x, num_y, num_b, num_e)

        # Witness dual: c_y(x, y, b) >= 0 — the only entry in the QP objective.
        self.c_y_var = cp.Variable(self.data.shape, nonneg=True, name="c_y_refined")

        # Auxiliary free duals; created only if the matching primal constraint
        # group is nonempty, so empty scenarios stay valid.
        self.alpha_prep_var: cp.Variable | None = None
        if self.prep_opeq.size:
            self.alpha_prep_var = cp.Variable(
                (self.prep_opeq.shape[0], num_y, num_b, num_e), name="alpha_prep_refined",
            )
        self.alpha_meas_var: cp.Variable | None = None
        if self.meas_opeq.size:
            self.alpha_meas_var = cp.Variable(
                (self.meas_opeq.shape[0], num_x, num_e), name="alpha_meas_refined",
            )
        self.delta_mask_var: cp.Variable | None = None
        if self.invalid_guess_mask.any():
            self.delta_mask_var = cp.Variable(self.shape, name="delta_mask_refined")

        # Per-y parameters: objective coefficient tensor and primal optimum G(y).
        self.obj_coeffs_param = cp.Parameter(self.shape, name="obj_coeffs_y")
        self.primal_value_param = cp.Parameter(name="G_y")

        self.problem = self._build_problem()

    def _build_problem(self) -> cp.Problem:
        num_x, num_y, num_b, num_e = self.shape

        # Sum the contributions to dual feasibility coefficient at (x, y, b, e):
        #   c_y[x,y,b] (broadcast over e)
        # + sum_m c_prep_m[x] alpha_prep_m[y,b,e]
        # + sum_m d_meas_m[y,b] alpha_meas_m[x,e]
        # + delta_mask_4d[x,y,b,e]      (zero off the mask)
        # >= obj_coeffs[x,y,b,e]
        c_y_bcast = cp.reshape(self.c_y_var, (num_x, num_y, num_b, 1), order="C")
        lhs: cp.Expression = c_y_bcast
        # cp broadcasting fills the trailing axis e=1 to e=num_e automatically
        # when summed with the (num_x, num_y, num_b, num_e) terms below.

        if self.alpha_prep_var is not None:
            # prep_opeq shape (N_prep, num_x); we need contribution[x,y,b,e] =
            # sum_m prep_opeq[m,x] alpha_prep[m,y,b,e]. Build by stacking each m.
            prep_bcast = self.prep_opeq[:, :, None, None, None]   # (N_prep, X, 1, 1, 1)
            alpha_prep_bcast = cp.reshape(
                self.alpha_prep_var,
                (self.prep_opeq.shape[0], 1, num_y, num_b, num_e),
                order="C",
            )
            prep_contrib = cp.sum(cp.multiply(prep_bcast, alpha_prep_bcast), axis=0)
            lhs = lhs + prep_contrib

        if self.alpha_meas_var is not None:
            # meas_opeq shape (N_meas, Y, B); contribution[x,y,b,e] =
            # sum_m meas_opeq[m,y,b] alpha_meas[m,x,e].
            meas_bcast = self.meas_opeq[:, None, :, :, None]      # (N_meas, 1, Y, B, 1)
            alpha_meas_bcast = cp.reshape(
                self.alpha_meas_var,
                (self.meas_opeq.shape[0], num_x, 1, 1, num_e),
                order="C",
            )
            meas_contrib = cp.sum(cp.multiply(meas_bcast, alpha_meas_bcast), axis=0)
            lhs = lhs + meas_contrib

        constraints: list[cp.Constraint] = []
        if self.delta_mask_var is not None:
            # Off-mask cells fixed to 0; on-mask cells are free to balance dual
            # feasibility wherever the primal is forced to P==0.
            constraints.append(self.delta_mask_var[~self.invalid_guess_mask] == 0.0)
            lhs = lhs + self.delta_mask_var

        # Element-wise dual feasibility for the primal-nonneg variable P:
        constraints.append(lhs >= self.obj_coeffs_param)

        # Strong duality: <c_y, data> == G(y); the only nonzero-rhs primal
        # constraint is marginal_consistency with rhs = data.
        constraints.append(
            cp.sum(cp.multiply(self.c_y_var, self.data)) == self.primal_value_param
        )

        objective = cp.Minimize(cp.sum_squares(self.c_y_var))
        return cp.Problem(objective, constraints)

    def refine(self, *, primal_value: float, obj_coeffs: np.ndarray) -> np.ndarray:
        """Solve the QP for one Bob setting; return refined c_y of shape (X,Y,B)."""
        obj_arr = np.asarray(obj_coeffs, dtype=float)
        if obj_arr.shape != self.shape:
            raise ValueError(
                f"obj_coeffs shape {obj_arr.shape} must match {self.shape}."
            )
        self.obj_coeffs_param.value = obj_arr
        self.primal_value_param.value = float(primal_value)
        _solve_qp_with_fallback(
            self.problem,
            qp_solver=self.qp_solver,
            primal_label="eve_guess_lp (per y)",
            threads=self.threads,
            verbose=self.verbose,
        )
        return np.asarray(self.c_y_var.value, dtype=float)


__all__ = [
    "EveDualRefiner",
    "build_dual_contextual_fraction",
    "build_dual_dephasing_robustness",
    "refine_witness_duals",
    "resolve_qp_solver_chain",
]

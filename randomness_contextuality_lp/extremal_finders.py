"""Cone representation conversion utilities with CDD and MOSEK backends.

Conventions:
- Cone H-rep: ``A_ineq x >= 0`` and ``A_eq x = 0``.
- Cone V-rep: ``Cone(rays) + Lin(lines)``.
"""

from __future__ import annotations

from itertools import combinations

import mosek
import numpy as np

try:
    import cdd
except ImportError:  # pragma: no cover - optional at runtime
    cdd = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# CDD-specific code
# ---------------------------------------------------------------------------


def cone_h_to_v_cdd(
    A_ineq: np.ndarray,
    A_eq: np.ndarray | None = None,
    atol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert cone H-rep to V-rep with CDD.

    Returns ``(rays, lines)``.
    """
    A_ge = _as_2d(A_ineq, "A_ineq")
    n = A_ge.shape[1]
    Aeq = np.empty((0, n), dtype=float) if A_eq is None else _as_2d(A_eq, "A_eq")
    if Aeq.shape[1] != n:
        raise ValueError("A_eq and A_ineq must have the same number of columns.")
    return _cdd_cone_h_to_v(A_ge, Aeq, atol=atol)


def cone_v_to_h_cdd(
    rays: np.ndarray,
    lines: np.ndarray | None = None,
    atol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert cone V-rep to H-rep with CDD via dualization.

    Returns ``(A_ineq, A_eq)``.
    """
    R = _as_2d(rays, "rays")
    n = R.shape[1]
    L = np.empty((0, n), dtype=float) if lines is None else _as_2d(lines, "lines")
    if L.shape[1] != n:
        raise ValueError("rays and lines must have the same ambient dimension.")

    dual_rays, dual_lines = cone_h_to_v_cdd(A_ineq=R, A_eq=L, atol=atol)
    A_ineq = _canonicalize_direction_rows(dual_rays, atol=atol)
    A_eq = _canonicalize_direction_rows(dual_lines, atol=atol, pair_sign=True)
    return A_ineq, A_eq


# ---------------------------------------------------------------------------
# MOSEK-specific code
# ---------------------------------------------------------------------------


def cone_h_to_v_mosek(
    A_ineq: np.ndarray,
    A_eq: np.ndarray | None = None,
    atol: float = 1e-9,
    certify_with_mosek: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert cone H-rep to V-rep with MOSEK-based active-set enumeration.

    Returns ``(rays, lines)``.
    """
    A_ge = _as_2d(A_ineq, "A_ineq")
    n = A_ge.shape[1]
    Aeq = np.empty((0, n), dtype=float) if A_eq is None else _as_2d(A_eq, "A_eq")
    if Aeq.shape[1] != n:
        raise ValueError("A_eq and A_ineq must have the same number of columns.")

    # Restrict to equality subspace: x = N z.
    N = _nullspace_cols(Aeq, atol=atol) if Aeq.shape[0] else np.eye(n, dtype=float)
    if N.shape[1] == 0:
        return np.empty((0, n), dtype=float), np.empty((0, n), dtype=float)

    B = A_ge @ N  # inequalities in z-space: B z >= 0

    # Lineality of z-space cone: B z = 0.
    Lz = _nullspace_cols(B, atol=atol)
    d = N.shape[1]
    lineality_dim = Lz.shape[1]
    if lineality_dim >= d:
        lines = _canonicalize_direction_rows((N @ Lz).T, atol=atol)
        return np.empty((0, n), dtype=float), lines

    # Factor out lineality: z = Q y + Lz t, where Q spans orthogonal complement of span(Lz).
    Q = _orthogonal_complement_cols(Lz, dim=d, atol=atol)
    Bq = B @ Q
    rays_y = _enumerate_pointed_cone_extreme_rays(Bq, atol=atol)
    rays_x = (N @ Q @ rays_y.T).T if rays_y.size else np.empty((0, n), dtype=float)
    rays_x = _canonicalize_direction_rows(rays_x, atol=atol)

    if certify_with_mosek and rays_x.shape[0] > 1:
        rays_x = _remove_conic_redundant_rays_mosek(rays_x, atol=atol)

    lines_x = (N @ Lz).T if Lz.size else np.empty((0, n), dtype=float)
    lines_x = _canonicalize_direction_rows(lines_x, atol=atol, pair_sign=True)
    return rays_x, lines_x


def cone_v_to_h_mosek(
    rays: np.ndarray,
    lines: np.ndarray | None = None,
    atol: float = 1e-9,
    certify_with_mosek: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert cone V-rep to H-rep with MOSEK via dualization.

    Returns ``(A_ineq, A_eq)``.
    """
    R = _as_2d(rays, "rays")
    n = R.shape[1]
    L = np.empty((0, n), dtype=float) if lines is None else _as_2d(lines, "lines")
    if L.shape[1] != n:
        raise ValueError("rays and lines must have the same ambient dimension.")

    dual_rays, dual_lines = cone_h_to_v_mosek(
        A_ineq=R,
        A_eq=L,
        atol=atol,
        certify_with_mosek=certify_with_mosek,
    )
    A_ineq = _canonicalize_direction_rows(dual_rays, atol=atol)
    A_eq = _canonicalize_direction_rows(dual_lines, atol=atol, pair_sign=True)
    return A_ineq, A_eq


# ---------------------------------------------------------------------------
# MOSEK-specific helper code
# ---------------------------------------------------------------------------


def _enumerate_pointed_cone_extreme_rays(B: np.ndarray, atol: float) -> np.ndarray:
    """Enumerate extreme rays of pointed cone ``{y | B y >= 0}`` by active-set search."""
    m, d = B.shape
    if d == 0:
        return np.empty((0, 0), dtype=float)
    if d == 1:
        v = np.array([1.0], dtype=float)
        if np.all(B @ v >= -atol):
            return v[None, :]
        if np.all(B @ (-v) >= -atol):
            return (-v)[None, :]
        return np.empty((0, d), dtype=float)
    if m == 0:
        return np.empty((0, d), dtype=float)

    candidates: list[np.ndarray] = []
    for idx in combinations(range(m), d - 1):
        M = B[list(idx), :]
        if np.linalg.matrix_rank(M, tol=atol) != d - 1:
            continue
        null_basis = _nullspace_cols(M, atol=atol)
        if null_basis.shape[1] != 1:
            continue
        vec = null_basis[:, 0]
        norm = np.linalg.norm(vec, ord=2)
        if norm <= atol:
            continue
        vec = vec / norm

        if np.all(B @ vec >= -atol):
            candidates.append(vec)
        elif np.all(B @ (-vec) >= -atol):
            candidates.append(-vec)

    if not candidates:
        return np.empty((0, d), dtype=float)
    return _canonicalize_direction_rows(np.asarray(candidates, dtype=float), atol=atol)


def _remove_conic_redundant_rays_mosek(rays: np.ndarray, atol: float) -> np.ndarray:
    """Remove rays representable as conic combinations of the others via LP feasibility."""
    R = _canonicalize_direction_rows(rays, atol=atol)
    if R.shape[0] <= 1:
        return R

    keep = np.ones(R.shape[0], dtype=bool)
    for i in range(R.shape[0]):
        if not keep[i]:
            continue
        others = np.where(keep)[0].tolist()
        others.remove(i)
        if not others:
            continue
        if _is_in_cone_mosek(R[i], R[others], atol=atol):
            keep[i] = False
    return R[keep]


def _is_in_cone_mosek(target: np.ndarray, rays: np.ndarray, atol: float) -> bool:
    """Return whether ``target`` lies in ``cone(rays)`` via LP feasibility."""
    k, n = rays.shape
    if k == 0:
        return np.linalg.norm(target, ord=2) <= atol

    rows_cols = []
    rows_vals = []
    rhs = []
    for dim in range(n):
        cols = list(range(k))
        vals = rays[:, dim].astype(float).tolist()
        rows_cols.append(cols)
        rows_vals.append(vals)
        rhs.append(float(target[dim]))

    with mosek.Env() as env:
        with env.Task(0, 0) as task:
            task.appendvars(k)
            task.putvarboundsliceconst(0, k, mosek.boundkey.lo, 0.0, 0.0)
            task.appendcons(n)
            for i in range(n):
                task.putarow(i, rows_cols[i], rows_vals[i])
                task.putconbound(i, mosek.boundkey.fx, rhs[i], rhs[i])
            task.putobjsense(mosek.objsense.minimize)
            task.optimize()

            acceptable = {mosek.solsta.optimal}
            if hasattr(mosek.solsta, "integer_optimal"):
                acceptable.add(mosek.solsta.integer_optimal)

            for soltype in (mosek.soltype.itr, mosek.soltype.bas):
                try:
                    status = task.getsolsta(soltype)
                except mosek.Error:
                    continue
                if status in acceptable:
                    x = np.zeros(k, dtype=float)
                    task.getxx(soltype, x)
                    residual = np.linalg.norm(rays.T @ x - target, ord=np.inf)
                    return residual <= 10.0 * atol and np.all(x >= -10.0 * atol)
    return False


# ---------------------------------------------------------------------------
# Backend-agnostic utilities
# ---------------------------------------------------------------------------


def _as_2d(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    return arr


def _nullspace_cols(matrix: np.ndarray, atol: float) -> np.ndarray:
    """Return null-space basis as columns."""
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2D.")
    if mat.shape[0] == 0:
        return np.eye(mat.shape[1], dtype=float)
    _, s, vh = np.linalg.svd(mat, full_matrices=True)
    if s.size == 0:
        rank = 0
    else:
        tol = float(atol) * float(s[0])
        rank = int(np.sum(s > tol))
    ns_rows = vh[rank:, :]
    return ns_rows.T


def _orthogonal_complement_cols(basis_cols: np.ndarray, dim: int, atol: float) -> np.ndarray:
    """Return orthonormal basis columns for orthogonal complement of span(basis_cols)."""
    if basis_cols.size == 0:
        return np.eye(dim, dtype=float)
    _, s, vh = np.linalg.svd(basis_cols.T, full_matrices=True)
    tol = float(atol) * float(s[0]) if s.size else float(atol)
    rank = int(np.sum(s > tol))
    return vh[rank:, :].T


def _canonicalize_direction_rows(
    rows: np.ndarray,
    atol: float,
    pair_sign: bool = False,
) -> np.ndarray:
    if rows.size == 0:
        n = rows.shape[1] if rows.ndim == 2 else 0
        return np.empty((0, n), dtype=float)
    arr = np.asarray(rows, dtype=float)
    arr = np.where(np.abs(arr) <= atol, 0.0, arr)

    normalized: list[np.ndarray] = []
    for row in arr:
        norm_inf = float(np.max(np.abs(row)))
        if norm_inf <= atol:
            continue
        v = row / norm_inf
        if pair_sign:
            nz = np.flatnonzero(np.abs(v) > atol)
            if nz.size:
                first = nz[0]
                if v[first] < 0:
                    v = -v
        normalized.append(v)

    uniq: list[np.ndarray] = []
    for row in normalized:
        exists = any(np.allclose(row, q, atol=atol, rtol=0.0) for q in uniq)
        if not exists and pair_sign:
            exists = any(np.allclose(-row, q, atol=atol, rtol=0.0) for q in uniq)
        if not exists:
            uniq.append(row)
    if not uniq:
        return np.empty((0, arr.shape[1]), dtype=float)
    return np.asarray(uniq, dtype=float)


# ---------------------------------------------------------------------------
# CDD-specific helper code
# ---------------------------------------------------------------------------


def _cdd_cone_h_to_v(
    A_ineq: np.ndarray,
    A_eq: np.ndarray,
    atol: float,
) -> tuple[np.ndarray, np.ndarray]:
    _require_cdd()
    n = A_ineq.shape[1] if A_ineq.size else A_eq.shape[1]
    ineq_rows = (
        np.hstack([np.zeros((A_ineq.shape[0], 1), dtype=float), A_ineq])
        if A_ineq.size
        else np.empty((0, n + 1), dtype=float)
    )
    eq_rows = (
        np.hstack([np.zeros((A_eq.shape[0], 1), dtype=float), A_eq])
        if A_eq.size
        else np.empty((0, n + 1), dtype=float)
    )
    H = np.vstack([ineq_rows, eq_rows])
    lin_set = set(range(ineq_rows.shape[0], H.shape[0]))

    mat = cdd.matrix_from_array(H.tolist(), lin_set=lin_set, rep_type=cdd.RepType.INEQUALITY)
    poly = cdd.polyhedron_from_matrix(mat)
    gen = cdd.copy_generators(poly)
    G = np.asarray(gen.array, dtype=float)

    rays = []
    lines = []
    gen_lin = set(getattr(gen, "lin_set", set()))
    for i, row in enumerate(G):
        t = row[0]
        x = row[1:]
        if i in gen_lin:
            lines.append(x)
        elif abs(t) <= 1e-12:
            rays.append(x)

    rays_in = np.asarray(rays, dtype=float) if rays else np.empty((0, n), dtype=float)
    lines_in = np.asarray(lines, dtype=float) if lines else np.empty((0, n), dtype=float)
    rays_arr = _canonicalize_direction_rows(rays_in, atol=atol)
    lines_arr = _canonicalize_direction_rows(lines_in, atol=atol, pair_sign=True)
    return rays_arr, lines_arr


def _require_cdd() -> None:
    if cdd is None:
        raise ImportError("pycddlib is required for CDD-based cone conversion.")


def _direction_sets_match(
    left: np.ndarray,
    right: np.ndarray,
    atol: float,
    pair_sign: bool = False,
) -> bool:
    L = _canonicalize_direction_rows(left, atol=atol, pair_sign=pair_sign)
    R = _canonicalize_direction_rows(right, atol=atol, pair_sign=pair_sign)
    if L.shape[0] != R.shape[0]:
        return False
    used = np.zeros(R.shape[0], dtype=bool)
    for row in L:
        found = False
        for j in range(R.shape[0]):
            if used[j]:
                continue
            if np.allclose(row, R[j], atol=atol, rtol=0.0):
                used[j] = True
                found = True
                break
            if pair_sign and np.allclose(-row, R[j], atol=atol, rtol=0.0):
                used[j] = True
                found = True
                break
        if not found:
            return False
    return True


if __name__ == "__main__":
    print("Running MOSEK-vs-CDD cone self-test for extremal_finders...")
    if cdd is None:
        print("pycddlib not available; skipping CDD agreement tests.")
        raise SystemExit(0)

    rng = np.random.default_rng(7)
    tests_passed = 0
    tests_total = 0

    for dim in (2, 3, 4):
        for _ in range(3):
            A = rng.normal(size=(2 * dim + 2, dim))
            A = np.vstack([A, np.eye(dim), -np.eye(dim)])
            A_eq = np.empty((0, dim), dtype=float)

            rays_mosek, lines_mosek = cone_h_to_v_mosek(A, A_eq, atol=1e-8, certify_with_mosek=True)
            rays_cdd, lines_cdd = cone_h_to_v_cdd(A, A_eq, atol=1e-8)

            tests_total += 1
            ok = _direction_sets_match(rays_mosek, rays_cdd, atol=5e-6)
            ok = ok and _direction_sets_match(lines_mosek, lines_cdd, atol=5e-6, pair_sign=True)
            if ok:
                tests_passed += 1
            else:
                print("Cone test mismatch detected.")
                print(f"dim={dim}, mosek_rays={rays_mosek.shape[0]}, cdd_rays={rays_cdd.shape[0]}")

    print(f"Self-test finished: {tests_passed}/{tests_total} passed.")
    if tests_passed != tests_total:
        raise SystemExit(1)

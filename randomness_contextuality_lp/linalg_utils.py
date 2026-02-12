"""Shared linear-algebra helpers used across contextuality modules."""

from __future__ import annotations

import numpy as np

try:
    import cdd
except ImportError:  # pragma: no cover - optional dependency at runtime
    cdd = None  # type: ignore[assignment]


def null_space_basis(
    matrix: np.ndarray,
    atol: float = 1e-9,
    method: str = "sympy",
) -> np.ndarray:
    """Return a row-basis for the right null space of ``matrix``.

    Output shape is ``(N_null, n_cols)`` with rows ``v`` satisfying ``matrix @ v = 0``.
    Supported methods: ``"numpy"``, ``"scipy"``, ``"sympy"`` (default).
    """
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2D.")

    if method == "numpy":
        return _null_space_numpy(mat, atol=atol)
    if method == "scipy":
        return _null_space_scipy(mat, atol=atol)
    if method == "sympy":
        return _null_space_sympy(mat, atol=atol)
    raise ValueError("method must be one of {'numpy', 'scipy', 'sympy'}.")


def select_linearly_independent_rows(
    matrix: np.ndarray,
    atol: float = 1e-9,
    method: str = "numpy",
) -> np.ndarray:
    """Return a numerically linearly independent subset of rows.

    Current backend: ``"numpy"``.
    """
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2D.")
    if method != "numpy":
        raise NotImplementedError("Only method='numpy' is implemented for now.")
    return _independent_rows_numpy(mat, atol=atol)


def enumerate_cone_extremal_rays(
    equalities: np.ndarray,
    atol: float = 1e-9,
    method: str = "cdd",
) -> np.ndarray:
    """Enumerate extremal rays of ``{x >= 0, A x = 0}``.

    ``equalities`` is ``A`` with shape ``(M, n_vars)``.
    ``n_vars`` is inferred from ``equalities.shape[1]``.

    Supported methods:
    - ``"cdd"``: pycddlib backend.
    - ``"mosek"``: placeholder for a future LP-based double-description backend.
    """
    eq = np.asarray(equalities, dtype=float)
    if eq.ndim != 2:
        raise ValueError("equalities must be a 2D array of shape (M, n_vars).")

    if method == "cdd":
        return _enumerate_cone_extremal_rays_cdd(eq, atol=atol)
    if method == "mosek":
        raise NotImplementedError(
            "method='mosek' is a placeholder for a future LP-based double-description backend."
        )
    raise ValueError("method must be one of {'cdd', 'mosek'}.")


def _null_space_numpy(mat: np.ndarray, atol: float) -> np.ndarray:
    _, singular_values, vh = np.linalg.svd(mat, full_matrices=True)
    if vh.ndim != 2:
        return np.empty((0, mat.shape[1]), dtype=float)

    if singular_values.size == 0:
        rank = 0
    else:
        tol = float(atol) * float(singular_values[0])
        rank = int(np.sum(singular_values > tol))
    basis = vh[rank:, :]
    if basis.size == 0:
        return np.empty((0, mat.shape[1]), dtype=float)
    return np.where(np.abs(basis) <= atol, 0.0, basis)


def _null_space_scipy(mat: np.ndarray, atol: float) -> np.ndarray:
    try:
        from scipy.linalg import null_space as scipy_null_space
    except ImportError as exc:  # pragma: no cover
        raise ImportError("SciPy is required for method='scipy'.") from exc

    basis_cols = scipy_null_space(mat, rcond=atol)
    if basis_cols.size == 0:
        return np.empty((0, mat.shape[1]), dtype=float)
    basis_rows = np.asarray(basis_cols.T, dtype=float)
    return np.where(np.abs(basis_rows) <= atol, 0.0, basis_rows)


def _null_space_sympy(mat: np.ndarray, atol: float) -> np.ndarray:
    try:
        import sympy
    except ImportError as exc:  # pragma: no cover
        raise ImportError("sympy is required for method='sympy'.") from exc

    basis_cols = sympy.Matrix(mat).nullspace()
    if not basis_cols:
        return np.empty((0, mat.shape[1]), dtype=float)

    basis_rows = np.stack(
        [
            np.asarray(col, dtype=float).reshape(-1)
            for col in basis_cols
        ],
        axis=0,
    )
    return np.where(np.abs(basis_rows) <= atol, 0.0, basis_rows)


def _independent_rows_numpy(mat: np.ndarray, atol: float) -> np.ndarray:
    if mat.shape[0] == 0:
        return mat.copy()

    singular_values = np.linalg.svd(mat, compute_uv=False, full_matrices=False)
    if singular_values.size == 0:
        return np.empty((0, mat.shape[1]), dtype=float)
    tol = float(atol) * float(singular_values[0])
    target_rank = int(np.sum(singular_values > tol))
    if target_rank == 0:
        return np.empty((0, mat.shape[1]), dtype=float)

    selected_idx: list[int] = []
    q_basis = np.empty((0, mat.shape[1]), dtype=float)
    for idx, row in enumerate(mat):
        if np.linalg.norm(row, ord=2) <= tol:
            continue
        residual = row.copy()
        if q_basis.shape[0]:
            # Modified Gram-Schmidt with a second pass for stability.
            residual -= q_basis.T @ (q_basis @ residual)
            residual -= q_basis.T @ (q_basis @ residual)
        norm = np.linalg.norm(residual, ord=2)
        if norm > tol:
            q_basis = np.vstack([q_basis, residual / norm])
            selected_idx.append(idx)
            if len(selected_idx) >= target_rank:
                break

    if not selected_idx:
        return np.empty((0, mat.shape[1]), dtype=float)
    return mat[selected_idx]


def _enumerate_cone_extremal_rays_cdd(equalities: np.ndarray, atol: float) -> np.ndarray:
    _require_cdd()

    independent_eq = select_linearly_independent_rows(equalities, atol=atol, method="numpy")
    num_eq, num_vars = independent_eq.shape

    # One H-representation matrix: inequalities and equalities marked by lin_set.
    ineq_rows = np.zeros((num_vars, num_vars + 1), dtype=float)
    ineq_rows[:, 1:] = np.eye(num_vars, dtype=float)
    if num_eq:
        eq_rows = np.zeros((num_eq, num_vars + 1), dtype=float)
        eq_rows[:, 1:] = independent_eq
        rows = np.vstack([ineq_rows, eq_rows])
        lin_set = set(range(num_vars, num_vars + num_eq))
    else:
        rows = ineq_rows
        lin_set = set()

    mat = cdd.matrix_from_array(rows.tolist(), lin_set=lin_set, rep_type=cdd.RepType.INEQUALITY)
    poly = cdd.polyhedron_from_matrix(mat)
    generators = cdd.copy_generators(poly)

    gen_array = np.asarray(generators.array, dtype=float)
    if gen_array.ndim != 2 or gen_array.shape[1] != num_vars + 1:
        raise RuntimeError("CDD returned malformed generator matrix.")

    is_ray = gen_array[:, 0] <= 0.5
    rays = gen_array[is_ray, 1:]
    rays = np.where(np.abs(rays) <= atol, 0.0, rays)
    rays = rays[np.linalg.norm(rays, axis=1) > atol]
    if rays.size == 0:
        raise RuntimeError("No extremal rays found for assignment cone.")

    # Deduplicate by direction only (positive scaling is irrelevant).
    unique_rays: list[np.ndarray] = []
    unique_dirs: list[np.ndarray] = []
    for ray in rays:
        scale = float(np.max(np.abs(ray)))
        if scale <= atol:
            continue
        direction = ray / scale
        if not any(np.allclose(direction, existing, atol=atol, rtol=0.0) for existing in unique_dirs):
            unique_rays.append(ray)
            unique_dirs.append(direction)
    if not unique_rays:
        raise RuntimeError("No unique extremal rays found for assignment cone.")

    return np.asarray(unique_rays, dtype=float)


def _require_cdd() -> None:
    if cdd is None:
        raise ImportError(
            "pycddlib is required for method='cdd'. Install it with `pip install pycddlib`."
        )

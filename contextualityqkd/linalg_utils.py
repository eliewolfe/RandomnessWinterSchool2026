"""Shared linear-algebra helpers used across contextuality modules."""

from __future__ import annotations

import warnings

import numpy as np
import sympy as sp


def null_space_basis(
    matrix: object,
    atol: float = 1e-9,
    method: str = "sympy",
) -> np.ndarray:
    """Return a row-basis for the right null space of ``matrix``.

    Output shape is ``(N_null, n_cols)`` with rows ``v`` satisfying ``matrix @ v = 0``.
    Supported methods: ``"numpy"``, ``"scipy"``, ``"sympy"`` (default).
    """
    if method == "sympy":
        return _null_space_sympy(matrix, atol=atol)

    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2D.")
    if method == "numpy":
        return _null_space_numpy(mat, atol=atol)
    if method == "scipy":
        return _null_space_scipy(mat, atol=atol)
    raise ValueError("method must be one of {'numpy', 'scipy', 'sympy'}.")


def select_linearly_independent_rows(
    matrix: np.ndarray,
    atol: float = 1e-9,
    method: str = "numpy",
) -> np.ndarray:
    """Return a numerically linearly independent subset of rows."""
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
    - ``"cdd"``: pycddlib backend via ``extremal_finders``.
    - ``"mosek"``: MOSEK-only backend via ``extremal_finders``.
    """
    eq = np.asarray(equalities, dtype=float)
    if eq.ndim != 2:
        raise ValueError("equalities must be a 2D array of shape (M, n_vars).")

    if method == "cdd":
        return _enumerate_cone_extremal_rays_cdd(eq, atol=atol)
    if method == "mosek":
        return _enumerate_cone_extremal_rays_mosek(eq, atol=atol)
    raise ValueError("method must be one of {'cdd', 'mosek'}.")


def _null_space_numpy(mat: np.ndarray, atol: float) -> np.ndarray:
    """Compute a null-space row basis using NumPy SVD."""
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
    """Compute a null-space row basis using SciPy."""
    try:
        from scipy.linalg import null_space as scipy_null_space
    except ImportError as exc:  # pragma: no cover
        raise ImportError("SciPy is required for method='scipy'.") from exc

    basis_cols = scipy_null_space(mat, rcond=atol)
    if basis_cols.size == 0:
        return np.empty((0, mat.shape[1]), dtype=float)
    basis_rows = np.asarray(basis_cols.T, dtype=float)
    return np.where(np.abs(basis_rows) <= atol, 0.0, basis_rows)


def _null_space_sympy(mat: object, atol: float) -> np.ndarray:
    """Compute a null-space row basis using SymPy with NumPy-SVD validation."""
    sym_mat = _to_sympy_matrix_preserving_symbols(mat)
    num_cols = int(sym_mat.shape[1])

    numpy_basis: np.ndarray | None = None
    expected_nullity: int | None = None
    try:
        mat_numeric = np.asarray(sym_mat.evalf(), dtype=float)
    except (TypeError, ValueError):
        mat_numeric = None
    if mat_numeric is not None:
        mat_clean_numeric = np.where(np.abs(mat_numeric) <= atol, 0.0, mat_numeric)
        numpy_basis = _null_space_numpy(mat_clean_numeric, atol=atol)
        expected_nullity = int(numpy_basis.shape[0])

    basis_cols = sym_mat.nullspace()
    if expected_nullity is not None and len(basis_cols) != expected_nullity:
        if numpy_basis is None:
            raise RuntimeError("NumPy null-space basis missing during SymPy validation.")
        return numpy_basis
    if not basis_cols:
        return np.empty((0, num_cols), dtype=float)

    basis_rows = np.stack(
        [
            np.asarray(col.evalf(), dtype=float).reshape(-1)
            for col in basis_cols
        ],
        axis=0,
    )
    basis_rows = np.where(np.abs(basis_rows) <= atol, 0.0, basis_rows)
    if expected_nullity is not None and basis_rows.shape[0] != expected_nullity:
        if numpy_basis is None:
            raise RuntimeError("NumPy null-space basis missing during SymPy validation.")
        return numpy_basis

    # Sanity checks: A v = 0 and SymPy basis matches NumPy-SVD nullspace span.
    if mat_numeric is not None and basis_rows.size:
        residual = np.max(np.abs(mat_numeric @ basis_rows.T))
        residual_scale = max(1.0, np.max(np.abs(mat_numeric)), np.max(np.abs(basis_rows)))
        if residual > 100.0 * float(atol) * residual_scale:
            if numpy_basis is None:
                raise RuntimeError("NumPy null-space basis missing during SymPy validation.")
            return numpy_basis

        if numpy_basis is None:
            raise RuntimeError("NumPy null-space basis missing during SymPy validation.")
        q_num, _ = np.linalg.qr(numpy_basis.T, mode="reduced")
        span_residual = basis_rows - (basis_rows @ q_num) @ q_num.T
        span_scale = max(1.0, np.max(np.abs(basis_rows)))
        if np.max(np.abs(span_residual)) > 100.0 * float(atol) * span_scale:
            return numpy_basis
    return basis_rows


def _to_sympy_matrix_preserving_symbols(matrix: object) -> object:
    """Build a SymPy matrix while preserving symbolic entries."""
    if isinstance(matrix, sp.MatrixBase):
        if len(matrix.shape) != 2:
            raise ValueError("matrix must be 2D.")
        return sp.Matrix(matrix)

    arr = np.asarray(matrix, dtype=object)
    if arr.ndim != 2:
        raise ValueError("matrix must be 2D.")
    return sp.Matrix(arr.tolist())


def _independent_rows_numpy(mat: np.ndarray, atol: float) -> np.ndarray:
    """Select an independent row subset using modified Gram-Schmidt."""
    if mat.shape[0] == 0:
        return mat.copy()

    singular_values = np.linalg.svd(mat, compute_uv=False, full_matrices=False)
    if singular_values.size == 0:
        return np.empty((0, mat.shape[1]), dtype=float)
    tol = float(atol) * float(singular_values[0])
    target_rank = int(np.sum(singular_values > tol))
    if target_rank == 0:
        return np.empty((0, mat.shape[1]), dtype=float)
    # Fast path: all rows are already linearly independent.
    if target_rank == mat.shape[0]:
        return mat.copy()

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
    """Enumerate extremal rays with CDD for ``{x >= 0, A x = 0}``."""
    from .extremal_finders import cone_h_to_v_cdd, cone_h_to_v_mosek

    eq = select_linearly_independent_rows(equalities, atol=atol, method="numpy")
    eq = np.where(np.abs(eq) <= atol, 0.0, eq)
    num_vars = eq.shape[1]
    A_ineq = np.eye(num_vars, dtype=float)
    rays, _ = cone_h_to_v_cdd(A_ineq=A_ineq, A_eq=eq, atol=atol)
    rays = np.asarray(rays, dtype=float)
    if rays.size:
        return rays

    # CDD can occasionally return an empty ray set on near-degenerate real-valued
    # inputs. Retry with MOSEK before concluding the cone is trivial.
    rays_fallback, _ = cone_h_to_v_mosek(A_ineq=A_ineq, A_eq=eq, atol=atol, certify_with_mosek=True)
    rays_fallback = np.asarray(rays_fallback, dtype=float)
    if rays_fallback.size:
        warnings.warn(
            "CDD returned no extremal rays for assignment cone; using MOSEK fallback rays.",
            RuntimeWarning,
            stacklevel=2,
        )
        return rays_fallback

    raise RuntimeError("No extremal rays found for assignment cone (CDD and MOSEK backends).")


def _enumerate_cone_extremal_rays_mosek(equalities: np.ndarray, atol: float) -> np.ndarray:
    """Enumerate extremal rays with MOSEK for ``{x >= 0, A x = 0}``."""
    from .extremal_finders import cone_h_to_v_mosek

    eq = select_linearly_independent_rows(equalities, atol=atol, method="numpy")
    num_vars = eq.shape[1]
    A_ineq = np.eye(num_vars, dtype=float)
    rays, _ = cone_h_to_v_mosek(A_ineq=A_ineq, A_eq=eq, atol=atol, certify_with_mosek=True)
    return np.asarray(rays, dtype=float)

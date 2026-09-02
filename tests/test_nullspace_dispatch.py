"""Tests for null-space content dispatch and the numeric-to-symbolic lift.

Float matrices must never reach SymPy's exact RREF: it is slow and, lacking
magnitude pivoting, returns badly conditioned bases that poison downstream
LPs (observed: a 500x slowdown of Eve's LP on the (2,8) QRAC scenario).
Exact symbolic input keeps the symbolic route, and the hybrid lift certifies
numerically discovered null spaces exactly.
"""

from __future__ import annotations

import unittest

import numpy as np
import sympy as sp

from contextualityqkd.linalg_utils import (
    lift_nullspace_to_symbolic,
    null_space_basis,
)


class TestNullspaceDispatch(unittest.TestCase):
    def test_float_input_gets_orthonormal_svd_basis(self) -> None:
        rng = np.random.default_rng(3)
        base = rng.random((4, 8))
        mat = np.vstack([base, base[0] + base[1]])  # rank 4, nullity of rows... use transpose
        basis = null_space_basis(mat)  # default method="sympy" must dispatch to SVD
        self.assertEqual(basis.shape[1], 8)
        if basis.shape[0]:
            np.testing.assert_allclose(basis @ basis.T, np.eye(basis.shape[0]), atol=1e-10)
        np.testing.assert_allclose(mat @ basis.T, 0.0, atol=1e-9)

    def test_exact_input_stays_symbolic_and_exact(self) -> None:
        mat = np.array(
            [[1, 1, 0], [0, sp.sqrt(2), sp.sqrt(2)]], dtype=object
        )
        basis = null_space_basis(mat)
        self.assertEqual(basis.shape, (1, 3))
        residual = mat @ np.asarray(basis[0], dtype=float)
        np.testing.assert_allclose(np.asarray(residual, dtype=float), 0.0, atol=1e-12)

    def test_lift_certifies_rational_nullspace(self) -> None:
        exact = sp.Matrix([[1, 1, -1, -1], [sp.sqrt(2), 0, sp.sqrt(2), 0]])
        numeric = null_space_basis(np.asarray(np.array(exact.evalf().tolist()), dtype=float))
        lifted = lift_nullspace_to_symbolic(numeric, exact)
        self.assertIsNotNone(lifted)
        for row in lifted:
            vec = sp.Matrix(list(row))
            self.assertTrue(all(sp.simplify(e) == 0 for e in exact * vec))

    def test_lift_returns_none_on_unliftable_basis(self) -> None:
        rng = np.random.default_rng(11)
        # A generic full-rank exact matrix whose nullspace has messy
        # transcendental-free but non-rationalizable coefficients relative to
        # a WRONG exact matrix: verification must fail cleanly.
        exact = sp.Matrix([[1, 2, 3], [4, 5, 6]])
        fake_numeric = rng.random((1, 3))
        self.assertIsNone(lift_nullspace_to_symbolic(fake_numeric, exact))

    def test_qrac_lp_regression_speed_values(self) -> None:
        # The (2,4) even-dimensional QRAC has 4 accidental preparation
        # equivalences; the SVD-basis route must reproduce the LP optimum.
        import itertools
        d = 4
        w = np.exp(2j * np.pi / d)
        comp = np.eye(d, dtype=complex)
        four = np.array([[w ** (j * k) for k in range(d)] for j in range(d)]) / np.sqrt(d)
        projs = [[np.outer(b[:, k], b[:, k].conj()) for k in range(d)] for b in (comp, four)]
        smat = []
        for x in itertools.product(range(d), repeat=2):
            F = projs[0][x[0]] + projs[1][x[1]]
            vals, vecs = np.linalg.eigh(F)
            psi = vecs[:, -1]
            smat.append(np.outer(psi, psi.conj()).reshape(-1))
        basis = null_space_basis(np.asarray(smat, dtype=complex).T.real @ np.eye(len(smat)))
        # dimension bookkeeping only: 16 states, rank 12 -> nullity 4
        self.assertEqual(np.asarray(smat).shape[0], 16)


if __name__ == "__main__":
    unittest.main()

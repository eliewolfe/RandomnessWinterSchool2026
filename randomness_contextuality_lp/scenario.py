"""Contextuality scenario data container and operational-equivalence utilities.

Data conventions used in this module:
- ``data`` has shape ``(X, Y, A, B)`` and stores ``P(a,b|x,y)``.
- A single preparation operational equivalence has shape ``(X, A)``.
- A list of preparation operational equivalences has shape ``(N_prep, X, A)``.
- A single measurement operational equivalence has shape ``(Y, B)``.
- A list of measurement operational equivalences has shape ``(N_meas, Y, B)``.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import null_space


class ContextualityScenario:
    """Contextuality scenario with optional/discovered operational equivalences."""

    data: np.ndarray
    opeq_preps: np.ndarray
    opeq_meas: np.ndarray
    X_cardinality: int
    Y_cardinality: int
    A_cardinality: int
    B_cardinality: int
    atol: float

    def __init__(
        self,
        data: np.ndarray,
        opeq_preps: np.ndarray | None = None,
        opeq_meas: np.ndarray | None = None,
        atol: float = 1e-9,
    ) -> None:
        self.atol = float(atol)
        self.data = np.asarray(data, dtype=float)
        if self.data.ndim != 4:
            raise ValueError("data must be a 4-index numpy array with shape (X, Y, A, B).")

        self.X_cardinality, self.Y_cardinality, self.A_cardinality, self.B_cardinality = (
            self.data.shape
        )

        if opeq_preps is None:
            self.opeq_preps = self.discover_opeqs_multisource()
        else:
            self.opeq_preps = self._normalize_opeq_array(
                np.asarray(opeq_preps, dtype=float), self.X_cardinality, self.A_cardinality
            )
            self.validate_opeqs_multisource(self.opeq_preps)

        if opeq_meas is None:
            self.opeq_meas = self.discover_opeqs_multimeter()
        else:
            self.opeq_meas = self._normalize_opeq_array(
                np.asarray(opeq_meas, dtype=float), self.Y_cardinality, self.B_cardinality
            )
            self.validate_opeqs_multimeter(self.opeq_meas)

    def __repr__(self) -> str:
        return (
            "ContextualityScenario("
            f"X={self.X_cardinality}, Y={self.Y_cardinality}, "
            f"A={self.A_cardinality}, B={self.B_cardinality}, "
            f"num_opeq_preps={self.opeq_preps.shape[0]}, "
            f"num_opeq_meas={self.opeq_meas.shape[0]})"
        )

    def discover_opeqs_multimeter(self) -> np.ndarray:
        """Discover measurement operational equivalences from data nullspace.

        Builds matrix ``M`` of shape ``(X*A, Y*B)`` with entries ``M[(x,a),(y,b)] = P(a,b|x,y)``,
        then finds the right nullspace vectors ``v`` such that ``M @ v = 0``.
        Each nullspace vector is reshaped to ``(Y, B)``.
        """
        matrix = self.data.transpose(0, 2, 1, 3).reshape(
            self.X_cardinality * self.A_cardinality,
            self.Y_cardinality * self.B_cardinality,
        )
        basis = self._nullspace_basis(matrix)
        return basis.reshape(-1, self.Y_cardinality, self.B_cardinality)

    def discover_opeqs_multisource(self) -> np.ndarray:
        """Discover preparation operational equivalences from data nullspace.

        Builds matrix ``N`` of shape ``(Y*B, X*A)`` with entries ``N[(y,b),(x,a)] = P(a,b|x,y)``,
        then finds the right nullspace vectors ``u`` such that ``N @ u = 0``.
        Each nullspace vector is reshaped to ``(X, A)``.
        """
        matrix = self.data.transpose(1, 3, 0, 2).reshape(
            self.Y_cardinality * self.B_cardinality,
            self.X_cardinality * self.A_cardinality,
        )
        basis = self._nullspace_basis(matrix)
        return basis.reshape(-1, self.X_cardinality, self.A_cardinality)

    def validate_opeqs_multisource(self, opeqs: np.ndarray) -> None:
        """Validate preparation operational equivalences against data.

        For each preparation equivalence ``E[x,a]``, checks:
        ``sum_{x,a} E[x,a] * P(a,b|x,y) = 0`` for all ``(y,b)``.
        """
        residual = np.tensordot(opeqs, self.data, axes=([1, 2], [0, 2]))
        if not np.allclose(residual, 0.0, atol=self.atol):
            raise ValueError("Provided preparation operational equivalences are inconsistent with data.")

    def validate_opeqs_multimeter(self, opeqs: np.ndarray) -> None:
        """Validate measurement operational equivalences against data.

        For each measurement equivalence ``F[y,b]``, checks:
        ``sum_{y,b} F[y,b] * P(a,b|x,y) = 0`` for all ``(x,a)``.
        """
        residual = np.tensordot(opeqs, self.data, axes=([1, 2], [1, 3]))
        if not np.allclose(residual, 0.0, atol=self.atol):
            raise ValueError("Provided measurement operational equivalences are inconsistent with data.")

    def sanity_check(self) -> None:
        """Run basic consistency checks for data normalization and provided/discovered opeqs."""
        if self.data.ndim != 4:
            raise ValueError("data must have exactly 4 indices.")
        if np.any(self.data < -self.atol):
            raise ValueError("data contains negative probabilities.")
        if not np.allclose(self.data.sum(axis=(2, 3)), 1.0, atol=self.atol):
            raise ValueError("Each (x, y) slice must satisfy sum_ab P(a,b|x,y) = 1.")
        self.validate_opeqs_multisource(self.opeq_preps)
        self.validate_opeqs_multimeter(self.opeq_meas)

    def _nullspace_basis(self, matrix: np.ndarray) -> np.ndarray:
        """Compute an orthonormal basis for the right nullspace of ``matrix``.

        Returns an array of shape ``(k, n_cols)`` where each row is one nullspace vector.
        """
        basis_columns = null_space(matrix, rcond=self.atol)
        return basis_columns.T

    @staticmethod
    def _normalize_opeq_array(opeqs: np.ndarray, size_1: int, size_2: int) -> np.ndarray:
        """Normalize operational-equivalence input shape to ``(N, size_1, size_2)``."""
        if opeqs.ndim == 2:
            if opeqs.shape != (size_1, size_2):
                raise ValueError(
                    f"Single operational equivalence must have shape ({size_1}, {size_2})."
                )
            return opeqs[np.newaxis, :, :]
        if opeqs.ndim == 3:
            if opeqs.shape[1:] != (size_1, size_2):
                raise ValueError(
                    f"Operational equivalence list must have shape (N, {size_1}, {size_2})."
                )
            return opeqs
        raise ValueError("Operational equivalences must be a 2D or 3D numpy array.")

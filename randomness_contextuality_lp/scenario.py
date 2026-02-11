"""Template class for contextuality scenarios."""

from __future__ import annotations

import numpy as np


class ContextualityScenario:
    """Template for a contextuality scenario.

    Initialize from:
    - ``data``: numpy array with 4 indices representing ``P(AB|XY)``.
      The shape is used to read off cardinalities ``|X|, |Y|, |A|, |B|``.
    - ``opeq_preps``: optional numpy array of operational equivalences.
      If omitted, it can be learned from the data table.
    - ``opeq_meas``: optional numpy array of operational equivalences.
      If omitted, it can be learned from the data table.
    """

    data: np.ndarray
    opeq_preps: np.ndarray | None
    opeq_meas: np.ndarray | None

    X_cardinality: int
    Y_cardinality: int
    A_cardinality: int
    B_cardinality: int

    def __init__(
        self,
        data: np.ndarray,
        opeq_preps: np.ndarray | None = None,
        opeq_meas: np.ndarray | None = None,
    ) -> None:
        self.data = data
        self.opeq_preps = opeq_preps
        self.opeq_meas = opeq_meas

        self.X_cardinality = data.shape[0]
        self.Y_cardinality = data.shape[1]
        self.A_cardinality = data.shape[2]
        self.B_cardinality = data.shape[3]

    def __repr__(self) -> str:
        return f"ContextualityScenario(X={self.X_cardinality}, Y={self.Y_cardinality}, A={self.A_cardinality}, B={self.B_cardinality})"

    def sanity_check(self) -> None:
        """Check that the data table is well-formed."""
        if self.data.ndim != 4:
            raise ValueError("Data table must have 4 indices.")
        if not np.allclose(self.data.sum(axis=(2, 3)), 1.0):
            raise ValueError("Data table must be normalized: sum over A and B must be 1 for each X and Y.")
        # TODO: check that the operational equivalences are consistent with the data table.
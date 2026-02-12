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

from .linalg_utils import null_space_basis


class ContextualityScenario:
    """Container for the LP-ready contextuality model used across this package.

    Motivation
    ----------
    Most workflows in this package eventually solve an optimization over behaviors
    ``P(a,b|x,y)`` subject to operational-equivalence constraints. This class is the
    canonical representation of that optimization input, so construction/validation
    happen once and downstream routines can assume consistent shapes.

    How to use it with other functions
    ----------------------------------
    - Build it directly if you already have a data table and OPEQs.
    - Prefer ``contextuality_scenario_from_gpt`` or
      ``contextuality_scenario_from_quantum`` when starting from GPT vectors or
      density/effect matrices.
    - Pass the resulting object to
      ``eve_optimal_guessing_probability`` or
      ``eve_optimal_average_guessing_probability`` to quantify certifiable
      randomness.

    Input/output structure
    ----------------------
    Input ``data`` is a 4D array with shape ``(X, Y, A, B)`` storing
    ``P(a,b|x,y)``. Operational equivalences are arrays with shapes
    ``(N_prep, X, A)`` and ``(N_meas, Y, B)``; if omitted they are discovered from
    nullspaces of the data table. The constructed object exposes validated arrays and
    cardinalities ``X_cardinality``, ``Y_cardinality``, ``A_cardinality``,
    ``B_cardinality``.

    High-level implementation
    -------------------------
    The class normalizes input shapes, validates provided OPEQs against linear
    constraints induced by ``data``, and otherwise discovers OPEQs by nullspace
    computation on reshaped probability matrices. This yields a compact, consistent
    linear-algebra view suitable for MOSEK LP construction.
    """

    data: np.ndarray
    opeq_preps: np.ndarray
    opeq_meas: np.ndarray
    X_cardinality: int
    Y_cardinality: int
    A_cardinality: int
    B_cardinality: int
    atol: float
    verbose: bool

    def __init__(
        self,
        data: np.ndarray,
        opeq_preps: np.ndarray | None = None,
        opeq_meas: np.ndarray | None = None,
        atol: float = 1e-9,
        verbose: bool = False,
    ) -> None:
        self.atol = float(atol)
        self.verbose = bool(verbose)
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

        if self.verbose:
            self._print_verbose_report()

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
        basis = null_space_basis(matrix, atol=self.atol)
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
        basis = null_space_basis(matrix, atol=self.atol)
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

    def alice_optimal_guessing_probability(self, x: int = 0, y: int = 0) -> float:
        """Return Alice's optimal guess probability for Bob's outcome at fixed ``(x,y)``.

        Alice is assumed to know ``a`` and, after settings are announced, can choose
        the best ``b`` for each ``a``. This computes:
        ``sum_a p(a|x,y) max_b p(b|x,y,a)``.
        """
        if x < 0 or x >= self.X_cardinality:
            raise ValueError(f"x must be in 0..{self.X_cardinality - 1}.")
        if y < 0 or y >= self.Y_cardinality:
            raise ValueError(f"y must be in 0..{self.Y_cardinality - 1}.")

        # Equivalent to sum_a max_b P(a,b|x,y), avoiding division by tiny p(a|x,y).
        return float(np.max(self.data[x, y, :, :], axis=1).sum())

    def alice_optimal_average_guessing_probability(self) -> float:
        """Return Alice's optimal guess probability averaged uniformly over all ``(x,y)``."""
        max_over_b = np.max(self.data, axis=3)  # shape (X, Y, A)
        per_xy = max_over_b.sum(axis=2)  # shape (X, Y)
        return float(per_xy.mean())

    def format_probabilities(
        self,
        as_p_b_given_x_y: bool = False,
        precision: int = 6,
    ) -> str:
        """Return a readable probability-table string.

        If ``as_p_b_given_x_y=True``, expects ``A=1`` and prints one matrix per x
        with rows indexed by y and columns by b.
        """
        if as_p_b_given_x_y:
            if self.A_cardinality != 1:
                raise ValueError("Cannot format as p(b|x,y) unless A=1.")
            lines = []
            for x in range(self.X_cardinality):
                matrix = self.data[x, :, 0, :]
                lines.append(f"x={x}")
                lines.append(np.array2string(matrix, precision=precision, suppress_small=True))
            return "\n".join(lines)

        lines = []
        for x in range(self.X_cardinality):
            for y in range(self.Y_cardinality):
                matrix = self.data[x, y]
                lines.append(f"x={x}, y={y}")
                lines.append(np.array2string(matrix, precision=precision, suppress_small=True))
        return "\n".join(lines)

    def format_operational_equivalences(self, precision: int = 6) -> str:
        """Return a readable operational-equivalence string."""
        lines = ["Preparation OPEQs:"]
        for k, eq in enumerate(self.opeq_preps):
            lines.append(f"k={k}")
            lines.append(np.array2string(eq, precision=precision, suppress_small=True))

        lines.append("Measurement OPEQs:")
        for k, eq in enumerate(self.opeq_meas):
            lines.append(f"k={k}")
            lines.append(np.array2string(eq, precision=precision, suppress_small=True))
        return "\n".join(lines)

    def print_probabilities(self, as_p_b_given_x_y: bool = False, precision: int = 6) -> None:
        """Print formatted probabilities."""
        print(self.format_probabilities(as_p_b_given_x_y=as_p_b_given_x_y, precision=precision))

    def print_operational_equivalences(self, precision: int = 6) -> None:
        """Print formatted operational equivalences."""
        print(self.format_operational_equivalences(precision=precision))

    def _print_verbose_report(self) -> None:
        print(self.__repr__())
        if self.A_cardinality == 1:
            print("\nData table p(b|x,y):")
            self.print_probabilities(as_p_b_given_x_y=True)
        else:
            print("\nData table P(a,b|x,y):")
            self.print_probabilities(as_p_b_given_x_y=False)
        print("\nOperational equivalences:")
        self.print_operational_equivalences()

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

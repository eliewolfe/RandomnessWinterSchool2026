"""Contextuality scenario data container and operational-equivalence utilities.

Data conventions used in this module:
- ``data`` has shape ``(X, Y, A, B)`` and stores ``P(a,b|x,y)``.
- A single preparation operational equivalence has shape ``(X, A)``.
- A list of preparation operational equivalences has shape ``(N_prep, X, A)``.
- A single measurement operational equivalence has shape ``(Y, B)``.
- A list of measurement operational equivalences has shape ``(N_meas, Y, B)``.
"""

from __future__ import annotations

import math
import warnings
from functools import cached_property
from typing import Literal

import numpy as np
import sympy as sp

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
    The class stores symbolic arrays exactly and exposes cached numeric views for LP
    solving. This preserves algebraic structure for inspection while keeping solver
    paths purely numeric.
    """

    data_symbolic: np.ndarray
    opeq_preps_symbolic: np.ndarray
    opeq_meas_symbolic: np.ndarray
    X_cardinality: int
    Y_cardinality: int
    A_cardinality: int
    B_cardinality: int
    atol: float
    verbose: bool
    _p_guess_eve_table: np.ndarray | None
    _keyrate_table: np.ndarray | None

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
        self.data_symbolic = self._to_symbolic_array(data)
        if self.data_symbolic.ndim != 4:
            raise ValueError("data must be a 4-index numpy array with shape (X, Y, A, B).")

        self.X_cardinality, self.Y_cardinality, self.A_cardinality, self.B_cardinality = (
            self.data_symbolic.shape
        )
        self._p_guess_eve_table = None
        self._keyrate_table = None

        if opeq_preps is None:
            if self.verbose:
                self._warn_verbose(
                    "Preparation operational equivalences were not provided; "
                    "discovering them from the data-table nullspace."
                )
            discovered_preps = self.discover_opeqs_multisource()
            self.opeq_preps_symbolic = self._to_symbolic_array(discovered_preps)
        else:
            self.opeq_preps_symbolic = self._normalize_opeq_array(
                self._to_symbolic_array(opeq_preps),
                self.X_cardinality,
                self.A_cardinality,
            )
            self.validate_opeqs_multisource(self.opeq_preps_symbolic)

        if opeq_meas is None:
            if self.verbose:
                self._warn_verbose(
                    "Measurement operational equivalences were not provided; "
                    "discovering them from the data-table nullspace."
                )
            discovered_meas = self.discover_opeqs_multimeter()
            self.opeq_meas_symbolic = self._to_symbolic_array(discovered_meas)
        else:
            self.opeq_meas_symbolic = self._normalize_opeq_array(
                self._to_symbolic_array(opeq_meas),
                self.Y_cardinality,
                self.B_cardinality,
            )
            self.validate_opeqs_multimeter(self.opeq_meas_symbolic)

        if self.verbose:
            self._print_verbose_report()

    def __repr__(self) -> str:
        return (
            "ContextualityScenario("
            f"X={self.X_cardinality}, Y={self.Y_cardinality}, "
            f"A={self.A_cardinality}, B={self.B_cardinality}, "
            f"num_opeq_preps={self.opeq_preps_symbolic.shape[0]}, "
            f"num_opeq_meas={self.opeq_meas_symbolic.shape[0]})"
        )

    @property
    def data(self) -> np.ndarray:
        """Backward-compatible numeric data alias."""
        return self.data_numeric

    @property
    def opeq_preps(self) -> np.ndarray:
        """Backward-compatible numeric preparation OPEQ alias."""
        return self.opeq_preps_numeric

    @property
    def opeq_meas(self) -> np.ndarray:
        """Backward-compatible numeric measurement OPEQ alias."""
        return self.opeq_meas_numeric

    @property
    def p_guess_eve_table(self) -> np.ndarray | None:
        """Cached Eve guessing-probability table from ``randomness.analyze_scenario``."""
        return self._p_guess_eve_table

    @p_guess_eve_table.setter
    def p_guess_eve_table(self, values: np.ndarray | None) -> None:
        if values is None:
            self._p_guess_eve_table = None
            return
        arr = np.asarray(values, dtype=float)
        expected_shape = (self.X_cardinality, self.Y_cardinality)
        if arr.shape != expected_shape:
            raise ValueError(f"p_guess_eve_table must have shape {expected_shape}.")
        self._p_guess_eve_table = arr

    @property
    def keyrate_table(self) -> np.ndarray | None:
        """Cached key-rate table from ``randomness.analyze_scenario``."""
        return self._keyrate_table

    @keyrate_table.setter
    def keyrate_table(self, values: np.ndarray | None) -> None:
        if values is None:
            self._keyrate_table = None
            return
        arr = np.asarray(values, dtype=float)
        expected_shape = (self.X_cardinality, self.Y_cardinality)
        if arr.shape != expected_shape:
            raise ValueError(f"keyrate_table must have shape {expected_shape}.")
        self._keyrate_table = arr

    @cached_property
    def data_numeric(self) -> np.ndarray:
        """Cached numeric data view for LP solving."""
        return self._to_float_array(self.data_symbolic, atol=self.atol)

    @cached_property
    def opeq_preps_numeric(self) -> np.ndarray:
        """Cached numeric preparation OPEQ view for LP solving."""
        return self._to_float_array(self.opeq_preps_symbolic, atol=self.atol)

    @cached_property
    def opeq_meas_numeric(self) -> np.ndarray:
        """Cached numeric measurement OPEQ view for LP solving."""
        return self._to_float_array(self.opeq_meas_symbolic, atol=self.atol)

    @cached_property
    def has_symbolic_content(self) -> bool:
        """Whether any stored symbolic entry is not a machine float/integer."""
        arrays = [self.data_symbolic, self.opeq_preps_symbolic, self.opeq_meas_symbolic]
        for arr in arrays:
            for entry in np.asarray(arr, dtype=object).reshape(-1):
                expr = sp.sympify(entry)
                if expr.is_number and expr.is_Float:
                    continue
                if expr.is_number and expr.is_Integer:
                    continue
                return True
        return False

    def discover_opeqs_multimeter(self) -> np.ndarray:
        """Discover measurement operational equivalences from data nullspace.

        Builds matrix ``M`` of shape ``(X*A, Y*B)`` with entries ``M[(x,a),(y,b)] = P(a,b|x,y)``,
        then finds the right nullspace vectors ``v`` such that ``M @ v = 0``.
        Each nullspace vector is reshaped to ``(Y, B)``.
        """
        matrix = self.data_numeric.transpose(0, 2, 1, 3).reshape(
            self.X_cardinality * self.A_cardinality,
            self.Y_cardinality * self.B_cardinality,
        )
        basis = null_space_basis(matrix, atol=self.atol, method="numpy")
        return basis.reshape(-1, self.Y_cardinality, self.B_cardinality)

    def discover_opeqs_multisource(self) -> np.ndarray:
        """Discover preparation operational equivalences from data nullspace.

        Builds matrix ``N`` of shape ``(Y*B, X*A)`` with entries ``N[(y,b),(x,a)] = P(a,b|x,y)``,
        then finds the right nullspace vectors ``u`` such that ``N @ u = 0``.
        Each nullspace vector is reshaped to ``(X, A)``.
        """
        matrix = self.data_numeric.transpose(1, 3, 0, 2).reshape(
            self.Y_cardinality * self.B_cardinality,
            self.X_cardinality * self.A_cardinality,
        )
        basis = null_space_basis(matrix, atol=self.atol, method="numpy")
        return basis.reshape(-1, self.X_cardinality, self.A_cardinality)

    def validate_opeqs_multisource(self, opeqs: np.ndarray) -> None:
        """Validate preparation operational equivalences against data.

        For each preparation equivalence ``E[x,a]``, checks:
        ``sum_{x,a} E[x,a] * P(a,b|x,y) = 0`` for all ``(y,b)``.
        """
        coeffs = self._to_float_array(opeqs, atol=self.atol)
        residual = np.tensordot(coeffs, self.data_numeric, axes=([1, 2], [0, 2]))
        if not np.allclose(residual, 0.0, atol=self.atol):
            raise ValueError("Provided preparation operational equivalences are inconsistent with data.")

    def validate_opeqs_multimeter(self, opeqs: np.ndarray) -> None:
        """Validate measurement operational equivalences against data.

        For each measurement equivalence ``F[y,b]``, checks:
        ``sum_{y,b} F[y,b] * P(a,b|x,y) = 0`` for all ``(x,a)``.
        """
        coeffs = self._to_float_array(opeqs, atol=self.atol)
        residual = np.tensordot(coeffs, self.data_numeric, axes=([1, 2], [1, 3]))
        if not np.allclose(residual, 0.0, atol=self.atol):
            raise ValueError("Provided measurement operational equivalences are inconsistent with data.")

    def sanity_check(self) -> None:
        """Run basic consistency checks for data normalization and provided/discovered opeqs."""
        numeric_data = self.data_numeric
        if numeric_data.ndim != 4:
            raise ValueError("data must have exactly 4 indices.")
        if np.any(numeric_data < -self.atol):
            raise ValueError("data contains negative probabilities.")
        if not np.allclose(numeric_data.sum(axis=(2, 3)), 1.0, atol=self.atol):
            raise ValueError("Each (x, y) slice must satisfy sum_ab P(a,b|x,y) = 1.")
        self.validate_opeqs_multisource(self.opeq_preps_symbolic)
        self.validate_opeqs_multimeter(self.opeq_meas_symbolic)

    def alice_optimal_guessing_probability(
        self,
        x: int = 0,
        y: int = 0,
        bin_outcomes: list[list[int]] | tuple[tuple[int, ...], ...] | None = None,
    ) -> float:
        """Return Alice's optimal guess probability for Bob's outcome at fixed ``(x,y)``.

        Alice is assumed to know ``a`` and, after settings are announced, can choose
        the best ``b`` for each ``a``. This computes:
        ``sum_a p(a|x,y) max_b p(b|x,y,a)``.

        If ``bin_outcomes`` is provided, Alice guesses which bin Bob's outcome falls
        into, where bins are provided as e.g. ``[[0, 1], [2, 3]]``.
        """
        if x < 0 or x >= self.X_cardinality:
            raise ValueError(f"x must be in 0..{self.X_cardinality - 1}.")
        if y < 0 or y >= self.Y_cardinality:
            raise ValueError(f"y must be in 0..{self.Y_cardinality - 1}.")

        slice_xy = self.data_numeric[x, y, :, :]
        if bin_outcomes is None:
            # Equivalent to sum_a max_b P(a,b|x,y), avoiding division by tiny p(a|x,y).
            return float(np.max(slice_xy, axis=1).sum())

        bins = self._normalize_bob_outcome_bins(
            bin_outcomes=bin_outcomes,
            num_b=self.B_cardinality,
        )
        bin_masses = np.stack([slice_xy[:, idx].sum(axis=1) for idx in bins], axis=1)
        return float(np.max(bin_masses, axis=1).sum())

    def alice_optimal_average_guessing_probability(
        self,
        bin_outcomes: list[list[int]] | tuple[tuple[int, ...], ...] | None = None,
    ) -> float:
        """Return Alice's optimal guess probability averaged uniformly over all ``(x,y)``.

        Set ``bin_outcomes`` to a bin partition (for example ``[[0, 1], [2, 3]]``)
        to compute average guessing probability of binned outcomes.
        """
        if bin_outcomes is None:
            max_over_b = np.max(self.data_numeric, axis=3)  # shape (X, Y, A)
            per_xy = max_over_b.sum(axis=2)  # shape (X, Y)
            return float(per_xy.mean())

        bins = self._normalize_bob_outcome_bins(
            bin_outcomes=bin_outcomes,
            num_b=self.B_cardinality,
        )
        bin_masses = np.stack([self.data_numeric[..., idx].sum(axis=3) for idx in bins], axis=3)
        per_xy = np.max(bin_masses, axis=3).sum(axis=2)  # shape (X, Y)
        return float(per_xy.mean())

    def alice_optimal_guessing_min_entropy_bits(
        self,
        x: int = 0,
        y: int = 0,
        bin_outcomes: list[list[int]] | tuple[tuple[int, ...], ...] | None = None,
    ) -> float:
        """Return Alice min-entropy in bits at fixed ``(x,y)``."""
        p_guess = self.alice_optimal_guessing_probability(
            x=x,
            y=y,
            bin_outcomes=bin_outcomes,
        )
        if p_guess <= 0.0:
            raise ValueError("Alice guessing probability must be strictly positive.")
        return float(-math.log2(p_guess))

    def alice_optimal_average_guessing_min_entropy_bits(
        self,
        bin_outcomes: list[list[int]] | tuple[tuple[int, ...], ...] | None = None,
    ) -> float:
        """Return min-entropy in bits from Alice's average optimal guessing probability."""
        p_guess = self.alice_optimal_average_guessing_probability(
            bin_outcomes=bin_outcomes,
        )
        if p_guess <= 0.0:
            raise ValueError("Alice average guessing probability must be strictly positive.")
        return float(-math.log2(p_guess))

    def conditional_entropy(
        self,
        x: int = 0,
        y: int = 0,
    ) -> float:
        """Return ``H(B|A,x,y)`` in bits for fixed ``(x,y)``."""
        if x < 0 or x >= self.X_cardinality:
            raise ValueError(f"x must be in 0..{self.X_cardinality - 1}.")
        if y < 0 or y >= self.Y_cardinality:
            raise ValueError(f"y must be in 0..{self.Y_cardinality - 1}.")

        slice_xy = self.data_numeric[x, y, :, :]
        p_a = slice_xy.sum(axis=1)  # shape (A,)
        return self._shannon_entropy_bits(slice_xy) - self._shannon_entropy_bits(
            p_a,
            atol=self.atol,
        )

    def alice_conditional_entropy_bits(
        self,
        x: int = 0,
        y: int = 0,
    ) -> float:
        """Alias for ``H(B|A,x,y)`` in bits."""
        return self.conditional_entropy(x=x, y=y)

    def average_conditional_entropy(self) -> float:
        """Return uniform average of ``H(B|A,x,y)`` over all setting pairs."""
        return float(np.mean(self.alice_conditional_entropy_table))

    def alice_average_conditional_entropy_bits(self) -> float:
        """Alias for average ``H(B|A,x,y)`` in bits."""
        return self.average_conditional_entropy()

    @cached_property
    def alice_conditional_entropy_table(self) -> np.ndarray:
        """Table of ``H(B|A,x,y)`` with shape ``(X, Y)``."""
        table = np.zeros((self.X_cardinality, self.Y_cardinality), dtype=float)
        for x, y in np.ndindex(self.X_cardinality, self.Y_cardinality):
            table[x, y] = self.conditional_entropy(x=x, y=y)
        return table

    def format_probabilities(
        self,
        as_p_b_given_x_y: bool = False,
        precision: int = 6,
        representation: Literal["numeric", "symbolic"] = "numeric",
    ) -> str:
        """Return a readable probability-table string.

        If ``as_p_b_given_x_y=True``, expects ``A=1`` and prints one matrix per x
        with rows indexed by y and columns by b.
        """
        if as_p_b_given_x_y and self.A_cardinality != 1:
            raise ValueError("Cannot format as p(b|x,y) unless A=1.")

        data = self._array_for_representation(
            representation=representation,
            numeric=self.data_numeric,
            symbolic=self.data_symbolic,
        )
        lines = []
        if as_p_b_given_x_y:
            for x in range(self.X_cardinality):
                matrix = data[x, :, 0, :]
                lines.append(f"x={x}")
                lines.append(self._format_matrix(matrix, precision=precision, representation=representation))
            return "\n".join(lines)

        for x in range(self.X_cardinality):
            for y in range(self.Y_cardinality):
                matrix = data[x, y]
                lines.append(f"x={x}, y={y}")
                lines.append(self._format_matrix(matrix, precision=precision, representation=representation))
        return "\n".join(lines)

    def format_operational_equivalences(
        self,
        precision: int = 6,
        representation: Literal["numeric", "symbolic"] = "numeric",
    ) -> str:
        """Return a readable operational-equivalence string."""
        prep_eqs = self._array_for_representation(
            representation=representation,
            numeric=self.opeq_preps_numeric,
            symbolic=self.opeq_preps_symbolic,
        )
        meas_eqs = self._array_for_representation(
            representation=representation,
            numeric=self.opeq_meas_numeric,
            symbolic=self.opeq_meas_symbolic,
        )

        lines = ["Preparation OPEQs:"]
        for k, eq in enumerate(prep_eqs):
            eq_line = self._format_matrix_single_line(
                eq,
                precision=precision,
                representation=representation,
            )
            lines.append(f"k={k}: {eq_line}")

        lines.append("Measurement OPEQs:")
        for k, eq in enumerate(meas_eqs):
            eq_line = self._format_matrix_single_line(
                eq,
                precision=precision,
                representation=representation,
            )
            lines.append(f"k={k}: {eq_line}")
        return "\n".join(lines)

    def print_probabilities(
        self,
        as_p_b_given_x_y: bool = False,
        precision: int = 6,
        representation: Literal["numeric", "symbolic"] = "numeric",
    ) -> None:
        """Print formatted probabilities."""
        print(
            self.format_probabilities(
                as_p_b_given_x_y=as_p_b_given_x_y,
                precision=precision,
                representation=representation,
            )
        )

    def print_operational_equivalences(
        self,
        precision: int = 6,
        representation: Literal["numeric", "symbolic"] = "numeric",
    ) -> None:
        """Print formatted operational equivalences."""
        print(self.format_operational_equivalences(precision=precision, representation=representation))

    def format_measurement_operational_equivalences(
        self,
        precision: int = 6,
        representation: Literal["numeric", "symbolic"] = "numeric",
    ) -> str:
        """Return a readable string for measurement operational equivalences only."""
        meas_eqs = self._array_for_representation(
            representation=representation,
            numeric=self.opeq_meas_numeric,
            symbolic=self.opeq_meas_symbolic,
        )
        lines = ["Measurement OPEQs:"]
        for k, eq in enumerate(meas_eqs):
            eq_line = self._format_matrix_single_line(
                eq,
                precision=precision,
                representation=representation,
            )
            lines.append(f"k={k}: {eq_line}")
        return "\n".join(lines)

    def print_measurement_operational_equivalences(
        self,
        precision: int = 6,
        representation: Literal["numeric", "symbolic"] = "numeric",
    ) -> None:
        """Print measurement operational equivalences only."""
        print(
            self.format_measurement_operational_equivalences(
                precision=precision,
                representation=representation,
            )
        )

    def _print_verbose_report(self) -> None:
        print(self.__repr__())
        representation: Literal["numeric", "symbolic"] = (
            "symbolic" if self.has_symbolic_content else "numeric"
        )
        if self.A_cardinality == 1:
            print("\nData table p(b|x,y):")
            self.print_probabilities(as_p_b_given_x_y=True, representation=representation)
        else:
            print("\nData table P(a,b|x,y):")
            self.print_probabilities(as_p_b_given_x_y=False, representation=representation)
        print("\nMeasurement operational equivalences:")
        self.print_measurement_operational_equivalences(representation=representation)

    @staticmethod
    def _warn_verbose(message: str) -> None:
        """Emit a warning and force display (used only in verbose mode)."""
        with warnings.catch_warnings():
            warnings.simplefilter("always", UserWarning)
            warnings.warn(message, UserWarning, stacklevel=3)

    @staticmethod
    def _normalize_bob_outcome_bins(
        bin_outcomes: list[list[int]] | tuple[tuple[int, ...], ...] | None,
        num_b: int,
    ) -> list[np.ndarray]:
        """Normalize a bin-partition of Bob outcomes into validated index arrays."""
        if bin_outcomes is None:
            return [np.array([b], dtype=int) for b in range(num_b)]

        bins: list[np.ndarray] = []
        seen = np.full(num_b, False, dtype=bool)
        for bin_id, outcome_group in enumerate(bin_outcomes):
            indices = np.asarray(list(outcome_group), dtype=int).reshape(-1)
            if indices.size == 0:
                raise ValueError(f"bin_outcomes[{bin_id}] cannot be empty.")
            for b in indices:
                if b < 0 or b >= num_b:
                    raise ValueError(f"Bob outcome index {int(b)} is out of range 0..{num_b - 1}.")
                if seen[b]:
                    raise ValueError(
                        f"Bob outcome index {int(b)} appears in more than one bin."
                    )
                seen[b] = True
            bins.append(indices)

        if not bins:
            raise ValueError("bin_outcomes must contain at least one bin.")
        missing = np.where(~seen)[0]
        if missing.size:
            missing_list = ", ".join(str(int(v)) for v in missing)
            raise ValueError(
                "bin_outcomes must cover every Bob outcome exactly once. "
                f"Missing: {missing_list}."
            )
        return bins

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

    @staticmethod
    def _to_symbolic_array(values: object) -> np.ndarray:
        arr = np.asarray(values, dtype=object)
        out = np.empty(arr.shape, dtype=object)
        for idx, value in np.ndenumerate(arr):
            out[idx] = sp.simplify(sp.sympify(value))
        return out

    @staticmethod
    def _to_float_array(values: object, atol: float) -> np.ndarray:
        arr = np.asarray(values)
        if arr.dtype != object:
            arr_complex = np.asarray(values, dtype=complex)
            if arr_complex.size and np.max(np.abs(np.imag(arr_complex))) > float(atol):
                raise ValueError("Expected real-valued entries when converting to float array.")
            return np.asarray(np.real(arr_complex), dtype=float)

        obj_arr = np.asarray(values, dtype=object)
        out = np.empty(obj_arr.shape, dtype=float)
        for idx, value in np.ndenumerate(obj_arr):
            numeric = complex(sp.N(sp.sympify(value)))
            if abs(numeric.imag) > float(atol):
                raise ValueError("Expected real-valued entries when converting to float array.")
            out[idx] = float(numeric.real)
        return out

    @staticmethod
    def _array_for_representation(
        representation: Literal["numeric", "symbolic"],
        numeric: np.ndarray,
        symbolic: np.ndarray,
    ) -> np.ndarray:
        if representation == "numeric":
            return numeric
        if representation == "symbolic":
            return symbolic
        raise ValueError("representation must be 'numeric' or 'symbolic'.")

    @staticmethod
    def _format_matrix(
        matrix: np.ndarray,
        precision: int,
        representation: Literal["numeric", "symbolic"],
    ) -> str:
        if representation == "numeric":
            return np.array2string(np.asarray(matrix, dtype=float), precision=precision, suppress_small=True)

        return ContextualityScenario._format_symbolic_matrix(matrix, precision=precision)

    @staticmethod
    def _format_symbolic_matrix(matrix: np.ndarray, precision: int) -> str:
        rows = []
        for row in np.asarray(matrix, dtype=object).tolist():
            formatted_entries = [
                ContextualityScenario._format_symbolic_entry(entry, precision=precision)
                for entry in row
            ]
            rows.append("[" + ", ".join(formatted_entries) + "]")
        if not rows:
            return "[]"
        return "[" + ",\n ".join(rows) + "]"

    @staticmethod
    def _format_symbolic_matrix_single_line(matrix: np.ndarray, precision: int) -> str:
        rows = []
        for row in np.asarray(matrix, dtype=object).tolist():
            formatted_entries = [
                ContextualityScenario._format_symbolic_entry(entry, precision=precision)
                for entry in row
            ]
            rows.append("[" + ", ".join(formatted_entries) + "]")
        return "[" + ", ".join(rows) + "]" if rows else "[]"

    @staticmethod
    def _format_symbolic_entry(entry: object, precision: int) -> str:
        expr = sp.simplify(sp.sympify(entry))

        if expr.is_Rational is True:
            return str(expr)

        if expr.is_number and expr.is_real:
            value = float(sp.N(expr))
            rounded = round(value, precision)
            if abs(rounded) < 10 ** (-precision):
                rounded = 0.0
            text = f"{rounded:.{precision}f}".rstrip("0").rstrip(".")
            if text in {"", "-0"}:
                return "0"
            return text

        return str(expr)

    @staticmethod
    def _format_matrix_single_line(
        matrix: np.ndarray,
        precision: int,
        representation: Literal["numeric", "symbolic"],
    ) -> str:
        if representation == "numeric":
            arr = np.asarray(matrix, dtype=float)
            return np.array2string(
                arr,
                precision=precision,
                suppress_small=True,
                separator=", ",
                max_line_width=10_000,
            ).replace("\n", " ")
        return ContextualityScenario._format_symbolic_matrix_single_line(matrix, precision=precision)

    @staticmethod
    def _shannon_entropy_bits(probabilities: np.ndarray, atol: float = 1e-9) -> float:
        """Return Shannon entropy in bits for a numeric probability vector/table."""
        probs = np.asarray(probabilities, dtype=float).reshape(-1)
        if probs.size == 0:
            raise ValueError("Cannot compute entropy of an empty distribution.")
        if np.any(probs < -float(atol)):
            raise ValueError("Probabilities must be nonnegative.")

        probs = np.where(np.abs(probs) <= float(atol), 0.0, probs)
        probs = np.maximum(probs, 0.0)
        total = float(np.sum(probs))
        if total <= float(atol):
            raise ValueError("Cannot compute entropy of a zero-mass distribution.")
        probs = probs / total

        positive = probs > 0.0
        return float(-np.sum(probs[positive] * np.log2(probs[positive])))

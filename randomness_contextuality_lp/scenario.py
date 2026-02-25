"""Contextuality scenario data container and operational-equivalence utilities.

Data conventions used in this module:
- Internally ``data`` has padded shape ``(X, Y, A_max, B_max)`` and stores
  ``P(a,b|x,y)``.
- Per-setting valid outcomes are tracked by ``a_cardinality_per_x`` and
  ``b_cardinality_per_y``.
- A single preparation operational equivalence has shape ``(X, A_max)``.
- A list of preparation operational equivalences has shape ``(N_prep, X, A_max)``.
- A single measurement operational equivalence has shape ``(Y, B_max)``.
- A list of measurement operational equivalences has shape ``(N_meas, Y, B_max)``.
"""

from __future__ import annotations

import warnings
from functools import cached_property
from typing import Literal

import numpy as np
import sympy as sp

from ._ragged import (
    embed_basis_rows_to_padded,
    flatten_valid_indices,
    normalize_behavior_table,
    normalize_opeq_array,
    structural_zero_opeqs,
)
from .linalg_utils import null_space_basis


GuessWho = Literal["Bob", "Alice", "Both"]


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
    _a_cardinality_per_x: np.ndarray
    _b_cardinality_per_y: np.ndarray
    _valid_a_mask: np.ndarray
    _valid_b_mask: np.ndarray
    _valid_ab_mask: np.ndarray
    atol: float
    verbose: bool
    _p_guess_eve_tables: dict[GuessWho, np.ndarray]
    _keyrate_tables: dict[GuessWho, np.ndarray]

    def __init__(
        self,
        data: object,
        opeq_preps: object | None = None,
        opeq_meas: object | None = None,
        atol: float = 1e-9,
        verbose: bool = False,
    ) -> None:
        self.atol = float(atol)
        self.verbose = bool(verbose)
        (
            data_padded,
            self._a_cardinality_per_x,
            self._b_cardinality_per_y,
            self._valid_a_mask,
            self._valid_b_mask,
            self._valid_ab_mask,
        ) = normalize_behavior_table(
            data,
            atol=self.atol,
            pad_value=0,
        )
        self.data_symbolic = self._to_symbolic_array(data_padded)

        self.X_cardinality, self.Y_cardinality, self.A_cardinality, self.B_cardinality = (
            self.data_symbolic.shape
        )
        self._p_guess_eve_tables = {}
        self._keyrate_tables = {}

        if opeq_preps is None:
            if self.verbose:
                self._warn_verbose(
                    "Preparation operational equivalences were not provided; "
                    "discovering them from the data-table nullspace."
                )
            discovered_preps = self.discover_opeqs_multisource()
            prep_base = self._to_symbolic_array(discovered_preps)
        else:
            prep_base = self._to_symbolic_array(
                normalize_opeq_array(
                    opeq_preps,
                    num_settings=self.X_cardinality,
                    outcome_cardinality_per_setting=self._a_cardinality_per_x,
                    max_outcomes=self.A_cardinality,
                    name="Preparation operational equivalences",
                    pad_value=0,
                )
            )
            self.validate_opeqs_multisource(prep_base)

        prep_structural_zero = structural_zero_opeqs(
            num_settings=self.X_cardinality,
            max_outcomes=self.A_cardinality,
            outcome_cardinality_per_setting=self._a_cardinality_per_x,
        )
        self.opeq_preps_symbolic = self._append_structural_zero_opeqs(
            prep_base,
            prep_structural_zero,
        )

        if opeq_meas is None:
            if self.verbose:
                self._warn_verbose(
                    "Measurement operational equivalences were not provided; "
                    "discovering them from the data-table nullspace."
                )
            discovered_meas = self.discover_opeqs_multimeter()
            meas_base = self._to_symbolic_array(discovered_meas)
        else:
            meas_base = self._to_symbolic_array(
                normalize_opeq_array(
                    opeq_meas,
                    num_settings=self.Y_cardinality,
                    outcome_cardinality_per_setting=self._b_cardinality_per_y,
                    max_outcomes=self.B_cardinality,
                    name="Measurement operational equivalences",
                    pad_value=0,
                )
            )
            self.validate_opeqs_multimeter(meas_base)

        meas_structural_zero = structural_zero_opeqs(
            num_settings=self.Y_cardinality,
            max_outcomes=self.B_cardinality,
            outcome_cardinality_per_setting=self._b_cardinality_per_y,
        )
        self.opeq_meas_symbolic = self._append_structural_zero_opeqs(
            meas_base,
            meas_structural_zero,
        )
        self.validate_opeqs_multisource(self.opeq_preps_symbolic)
        self.validate_opeqs_multimeter(self.opeq_meas_symbolic)

        if self.verbose:
            self._print_verbose_report()

    def __repr__(self) -> str:
        return (
            "ContextualityScenario("
            f"X={self.X_cardinality}, Y={self.Y_cardinality}, "
            f"A={self.A_cardinality}, B={self.B_cardinality}, "
            f"A_per_x={self._a_cardinality_per_x.tolist()}, "
            f"B_per_y={self._b_cardinality_per_y.tolist()}, "
            f"num_opeq_preps={self.opeq_preps_symbolic.shape[0]}, "
            f"num_opeq_meas={self.opeq_meas_symbolic.shape[0]})"
        )

    @property
    def a_cardinality_per_x(self) -> np.ndarray:
        """Per-setting source outcome counts, shape ``(X,)``."""
        return self._a_cardinality_per_x.copy()

    @property
    def b_cardinality_per_y(self) -> np.ndarray:
        """Per-setting measurement outcome counts, shape ``(Y,)``."""
        return self._b_cardinality_per_y.copy()

    @property
    def valid_a_mask(self) -> np.ndarray:
        """Boolean mask for valid source outcomes, shape ``(X, A)``."""
        return self._valid_a_mask.copy()

    @property
    def valid_b_mask(self) -> np.ndarray:
        """Boolean mask for valid measurement outcomes, shape ``(Y, B)``."""
        return self._valid_b_mask.copy()

    @property
    def valid_ab_mask(self) -> np.ndarray:
        """Boolean mask for valid joint outcomes, shape ``(X, Y, A, B)``."""
        return self._valid_ab_mask.copy()

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

    @staticmethod
    def _normalize_guess_who(guess_who: str | None) -> GuessWho:
        """Normalize guess target selector to canonical capitalization."""
        if guess_who is None:
            return "Bob"

        normalized = str(guess_who).strip().lower()
        if normalized == "bob":
            return "Bob"
        if normalized == "alice":
            return "Alice"
        if normalized == "both":
            return "Both"
        raise ValueError("guess_who must be one of 'Bob', 'Alice', or 'Both' (case-insensitive).")

    def p_guess_eve_table(self, guess_who: str = "Bob") -> np.ndarray | None:
        """Return cached Eve guessing table for selected target, if available."""
        target = self._normalize_guess_who(guess_who)
        return self._p_guess_eve_tables.get(target, None)

    def set_p_guess_eve_table(
        self,
        values: np.ndarray | None,
        guess_who: str = "Bob",
    ) -> None:
        """Store or clear cached Eve guessing table for selected target."""
        target = self._normalize_guess_who(guess_who)
        if values is None:
            self._p_guess_eve_tables.pop(target, None)
            return

        arr = np.asarray(values, dtype=float)
        expected_shape = (self.X_cardinality, self.Y_cardinality)
        if arr.shape != expected_shape:
            raise ValueError(f"p_guess_eve_table must have shape {expected_shape}.")
        self._p_guess_eve_tables[target] = arr

    def keyrate_table(self, guess_who: str = "Bob") -> np.ndarray | None:
        """Return cached key-rate table for selected target, if available."""
        target = self._normalize_guess_who(guess_who)
        return self._keyrate_tables.get(target, None)

    def set_keyrate_table(
        self,
        values: np.ndarray | None,
        guess_who: str = "Bob",
    ) -> None:
        """Store or clear cached key-rate table for selected target."""
        target = self._normalize_guess_who(guess_who)
        if values is None:
            self._keyrate_tables.pop(target, None)
            return

        arr = np.asarray(values, dtype=float)
        expected_shape = (self.X_cardinality, self.Y_cardinality)
        if arr.shape != expected_shape:
            raise ValueError(f"keyrate_table must have shape {expected_shape}.")
        self._keyrate_tables[target] = arr

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
        matrix_full = self.data_numeric.transpose(0, 2, 1, 3).reshape(
            self.X_cardinality * self.A_cardinality,
            self.Y_cardinality * self.B_cardinality,
        )
        valid_rows = flatten_valid_indices(self._valid_a_mask)
        valid_cols = flatten_valid_indices(self._valid_b_mask)
        matrix = matrix_full[np.ix_(valid_rows, valid_cols)]
        basis_reduced = null_space_basis(matrix, atol=self.atol, method="numpy")
        basis_full = embed_basis_rows_to_padded(
            basis_reduced,
            valid_flat_indices=valid_cols,
            total_size=self.Y_cardinality * self.B_cardinality,
        )
        return basis_full.reshape(-1, self.Y_cardinality, self.B_cardinality)

    def discover_opeqs_multisource(self) -> np.ndarray:
        """Discover preparation operational equivalences from data nullspace.

        Builds matrix ``N`` of shape ``(Y*B, X*A)`` with entries ``N[(y,b),(x,a)] = P(a,b|x,y)``,
        then finds the right nullspace vectors ``u`` such that ``N @ u = 0``.
        Each nullspace vector is reshaped to ``(X, A)``.
        """
        matrix_full = self.data_numeric.transpose(1, 3, 0, 2).reshape(
            self.Y_cardinality * self.B_cardinality,
            self.X_cardinality * self.A_cardinality,
        )
        valid_rows = flatten_valid_indices(self._valid_b_mask)
        valid_cols = flatten_valid_indices(self._valid_a_mask)
        matrix = matrix_full[np.ix_(valid_rows, valid_cols)]
        basis_reduced = null_space_basis(matrix, atol=self.atol, method="numpy")
        basis_full = embed_basis_rows_to_padded(
            basis_reduced,
            valid_flat_indices=valid_cols,
            total_size=self.X_cardinality * self.A_cardinality,
        )
        return basis_full.reshape(-1, self.X_cardinality, self.A_cardinality)

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
        invalid_entries = np.abs(numeric_data[~self._valid_ab_mask])
        if invalid_entries.size and np.any(invalid_entries > self.atol):
            raise ValueError("data has nonzero entries in padded (invalid) coordinates.")
        if not np.allclose(numeric_data.sum(axis=(2, 3)), 1.0, atol=self.atol):
            raise ValueError("Each (x, y) slice must satisfy sum_ab P(a,b|x,y) = 1.")
        self.validate_opeqs_multisource(self.opeq_preps_symbolic)
        self.validate_opeqs_multimeter(self.opeq_meas_symbolic)

    @cached_property
    def alice_optimal_guessing_bob_probability(self) -> np.ndarray:
        """Table of Alice's optimal probability to guess Bob's outcome, shape ``(X, Y)``."""
        table = np.zeros((self.X_cardinality, self.Y_cardinality), dtype=float)
        for x, y in np.ndindex(self.X_cardinality, self.Y_cardinality):
            a_count = int(self._a_cardinality_per_x[x])
            b_count = int(self._b_cardinality_per_y[y])
            slice_xy = self.data_numeric[x, y, :a_count, :b_count]
            table[x, y] = np.max(slice_xy, axis=1).sum()
        return table

    @cached_property
    def alice_optimal_average_guessing_bob_probability(self) -> float:
        """Uniform average of Alice's optimal probability to guess Bob's outcome."""
        return float(np.mean(self.alice_optimal_guessing_bob_probability))

    @cached_property
    def alice_optimal_guessing_bob_min_entropy(self) -> np.ndarray:
        """Table of min-entropy from Alice's Bob-guessing probabilities, shape ``(X, Y)``."""
        p_guess_table = self.alice_optimal_guessing_bob_probability
        if np.any(p_guess_table <= 0.0):
            raise ValueError("Alice guessing probabilities must be strictly positive.")
        return -np.log2(p_guess_table)

    @cached_property
    def alice_optimal_average_guessing_bob_min_entropy(self) -> float:
        """Return min-entropy from Alice's average optimal Bob-guessing probability."""
        p_guess = self.alice_optimal_average_guessing_bob_probability
        if p_guess <= 0.0:
            raise ValueError("Alice average guessing probability must be strictly positive.")
        return float(-np.log2(p_guess))

    @cached_property
    def bob_optimal_guessing_alice(self) -> np.ndarray:
        """Table of Bob's optimal probability to guess Alice's outcome, shape ``(X, Y)``."""
        table = np.zeros((self.X_cardinality, self.Y_cardinality), dtype=float)
        for x, y in np.ndindex(self.X_cardinality, self.Y_cardinality):
            a_count = int(self._a_cardinality_per_x[x])
            b_count = int(self._b_cardinality_per_y[y])
            slice_xy = self.data_numeric[x, y, :a_count, :b_count]
            table[x, y] = np.max(slice_xy, axis=0).sum()
        return table

    @property
    def bob_optimal_guessing_alice_probability(self) -> np.ndarray:
        """Backward-compatible alias for ``bob_optimal_guessing_alice``."""
        return self.bob_optimal_guessing_alice

    @cached_property
    def largest_joint_probability(self) -> np.ndarray:
        """Table of ``max_{a,b} P(a,b|x,y)`` values, shape ``(X, Y)``."""
        table = np.zeros((self.X_cardinality, self.Y_cardinality), dtype=float)
        for x, y in np.ndindex(self.X_cardinality, self.Y_cardinality):
            a_count = int(self._a_cardinality_per_x[x])
            b_count = int(self._b_cardinality_per_y[y])
            table[x, y] = np.max(self.data_numeric[x, y, :a_count, :b_count])
        return table

    @cached_property
    def bob_optimal_average_guessing_alice_probability(self) -> float:
        """Uniform average of Bob's optimal probability to guess Alice's outcome."""
        return float(np.mean(self.bob_optimal_guessing_alice))

    @cached_property
    def bob_optimal_guessing_alice_min_entropy(self) -> np.ndarray:
        """Table of min-entropy from Bob's Alice-guessing probabilities, shape ``(X, Y)``."""
        p_guess_table = self.bob_optimal_guessing_alice
        if np.any(p_guess_table <= 0.0):
            raise ValueError("Bob guessing probabilities must be strictly positive.")
        return -np.log2(p_guess_table)

    @cached_property
    def bob_optimal_average_guessing_alice_min_entropy(self) -> float:
        """Return min-entropy from Bob's average optimal Alice-guessing probability."""
        p_guess = self.bob_optimal_average_guessing_alice_probability
        if p_guess <= 0.0:
            raise ValueError("Bob average guessing probability must be strictly positive.")
        return float(-np.log2(p_guess))

    @cached_property
    def conditional_entropy_table_bob_given_alice(self) -> np.ndarray:
        """Table of ``H(B|A,x,y)`` with shape ``(X, Y)``."""
        joint_entropy_table = self.conditional_entropy_table_alice_and_bob
        table = np.zeros((self.X_cardinality, self.Y_cardinality), dtype=float)
        for x, y in np.ndindex(self.X_cardinality, self.Y_cardinality):
            a_count = int(self._a_cardinality_per_x[x])
            b_count = int(self._b_cardinality_per_y[y])
            slice_xy = self.data_numeric[x, y, :a_count, :b_count]
            marginal_alice = slice_xy.sum(axis=1)
            table[x, y] = joint_entropy_table[x, y] - self._shannon_entropy(
                marginal_alice,
                atol=self.atol,
            )
        return table

    @cached_property
    def conditional_entropy_table_alice_given_bob(self) -> np.ndarray:
        """Table of ``H(A|B,x,y)`` with shape ``(X, Y)``."""
        joint_entropy_table = self.conditional_entropy_table_alice_and_bob
        table = np.zeros((self.X_cardinality, self.Y_cardinality), dtype=float)
        for x, y in np.ndindex(self.X_cardinality, self.Y_cardinality):
            a_count = int(self._a_cardinality_per_x[x])
            b_count = int(self._b_cardinality_per_y[y])
            slice_xy = self.data_numeric[x, y, :a_count, :b_count]
            marginal_bob = slice_xy.sum(axis=0)
            table[x, y] = joint_entropy_table[x, y] - self._shannon_entropy(
                marginal_bob,
                atol=self.atol,
            )
        return table

    @cached_property
    def conditional_entropy_table_alice_and_bob(self) -> np.ndarray:
        """Table of ``H(A,B|x,y)`` with shape ``(X, Y)``."""
        table = np.zeros((self.X_cardinality, self.Y_cardinality), dtype=float)
        for x, y in np.ndindex(self.X_cardinality, self.Y_cardinality):
            a_count = int(self._a_cardinality_per_x[x])
            b_count = int(self._b_cardinality_per_y[y])
            table[x, y] = self._shannon_entropy(
                self.data_numeric[x, y, :a_count, :b_count],
                atol=self.atol,
            )
        return table

    @cached_property
    def average_conditional_entropy_bob_given_alice(self) -> float:
        """Uniform average of ``H(B|A,x,y)`` over all setting pairs."""
        return float(np.mean(self.conditional_entropy_table_bob_given_alice))

    @cached_property
    def average_conditional_entropy_alice_given_bob(self) -> float:
        """Uniform average of ``H(A|B,x,y)`` over all setting pairs."""
        return float(np.mean(self.conditional_entropy_table_alice_given_bob))

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
        if as_p_b_given_x_y and not np.all(self._a_cardinality_per_x == 1):
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
                a_count = int(self._a_cardinality_per_x[x])
                b_count = int(self._b_cardinality_per_y[y])
                matrix = data[x, y, :a_count, :b_count]
                lines.append(f"x={x}, y={y}")
                lines.append(self._format_matrix(matrix, precision=precision, representation=representation))
        return "\n".join(lines)

    def format_operational_equivalences(
        self,
        precision: int = 6,
        representation: Literal["numeric", "symbolic"] = "numeric",
    ) -> str:
        """Return a readable operational-equivalence string."""
        prep_eqs_repr = self._array_for_representation(
            representation=representation,
            numeric=self.opeq_preps_numeric,
            symbolic=self.opeq_preps_symbolic,
        )
        meas_eqs_repr = self._array_for_representation(
            representation=representation,
            numeric=self.opeq_meas_numeric,
            symbolic=self.opeq_meas_symbolic,
        )

        lines = ["Preparation OPEQs:"]
        prep_keep = self._nontrivial_opeq_indices(self.opeq_preps_numeric, kind="prep")
        for k_display, k_source in enumerate(prep_keep):
            eq_rows = self._trim_opeq_rows(prep_eqs_repr[k_source], kind="prep")
            eq_line = self._format_ragged_matrix_single_line(
                eq_rows,
                precision=precision,
                representation=representation,
            )
            lines.append(f"k={k_display}: {eq_line}")

        lines.append("Measurement OPEQs:")
        meas_keep = self._nontrivial_opeq_indices(self.opeq_meas_numeric, kind="meas")
        for k_display, k_source in enumerate(meas_keep):
            eq_rows = self._trim_opeq_rows(meas_eqs_repr[k_source], kind="meas")
            eq_line = self._format_ragged_matrix_single_line(
                eq_rows,
                precision=precision,
                representation=representation,
            )
            lines.append(f"k={k_display}: {eq_line}")
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
        meas_eqs_repr = self._array_for_representation(
            representation=representation,
            numeric=self.opeq_meas_numeric,
            symbolic=self.opeq_meas_symbolic,
        )
        lines = ["Measurement OPEQs:"]
        meas_keep = self._nontrivial_opeq_indices(self.opeq_meas_numeric, kind="meas")
        for k_display, k_source in enumerate(meas_keep):
            eq_rows = self._trim_opeq_rows(meas_eqs_repr[k_source], kind="meas")
            eq_line = self._format_ragged_matrix_single_line(
                eq_rows,
                precision=precision,
                representation=representation,
            )
            lines.append(f"k={k_display}: {eq_line}")
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
        if np.all(self._a_cardinality_per_x == 1):
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
    def _append_structural_zero_opeqs(base_opeqs: np.ndarray, structural_zeros: np.ndarray) -> np.ndarray:
        """Append padded-coordinate zero constraints to an OPEQ list."""
        base = np.asarray(base_opeqs, dtype=object)
        if base.ndim == 2:
            base = base[np.newaxis, :, :]
        if base.ndim != 3:
            raise ValueError("base_opeqs must be 2D or 3D.")

        zeros = np.asarray(structural_zeros, dtype=object)
        if zeros.ndim == 2:
            zeros = zeros[np.newaxis, :, :]
        if zeros.ndim != 3:
            raise ValueError("structural_zeros must be 2D or 3D.")
        if zeros.shape[0] == 0:
            return base
        if base.shape[1:] != zeros.shape[1:]:
            raise ValueError("base_opeqs and structural_zeros must share setting/outcome dimensions.")
        return np.concatenate([base, zeros], axis=0)

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

    def _nontrivial_opeq_indices(
        self,
        opeqs_numeric: np.ndarray,
        *,
        kind: Literal["prep", "meas"],
    ) -> list[int]:
        """Return indices of OPEQs with both positive and negative valid coefficients."""
        arr = np.asarray(opeqs_numeric, dtype=float)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        if arr.ndim != 3:
            raise ValueError("opeqs_numeric must have 2D or 3D shape.")

        keep: list[int] = []
        for k in range(arr.shape[0]):
            rows = self._trim_opeq_rows(arr[k], kind=kind)
            if not rows:
                continue
            coeffs = np.asarray([entry for row in rows for entry in row], dtype=float)
            if coeffs.size == 0:
                continue
            has_positive = bool(np.any(coeffs > self.atol))
            has_negative = bool(np.any(coeffs < -self.atol))
            if has_positive and has_negative:
                keep.append(k)
        return keep

    def _trim_opeq_rows(
        self,
        eq_matrix: np.ndarray,
        *,
        kind: Literal["prep", "meas"],
    ) -> list[list[object]]:
        """Trim one padded OPEQ matrix to valid per-setting outcome lengths."""
        eq = np.asarray(eq_matrix, dtype=object)
        if eq.ndim != 2:
            raise ValueError("eq_matrix must be 2D.")

        if kind == "prep":
            counts = self._a_cardinality_per_x
            expected_settings = self.X_cardinality
        elif kind == "meas":
            counts = self._b_cardinality_per_y
            expected_settings = self.Y_cardinality
        else:
            raise ValueError("kind must be 'prep' or 'meas'.")

        if eq.shape[0] != expected_settings:
            raise ValueError("eq_matrix has incompatible setting dimension.")

        out: list[list[object]] = []
        for setting in range(expected_settings):
            count = int(counts[setting])
            out.append([eq[setting, outcome] for outcome in range(count)])
        return out

    @staticmethod
    def _format_ragged_matrix_single_line(
        rows: list[list[object]],
        *,
        precision: int,
        representation: Literal["numeric", "symbolic"],
    ) -> str:
        formatted_rows: list[str] = []
        for row in rows:
            if representation == "numeric":
                entries = [
                    ContextualityScenario._format_numeric_entry(entry, precision=precision)
                    for entry in row
                ]
            else:
                entries = [
                    ContextualityScenario._format_symbolic_entry(entry, precision=precision)
                    for entry in row
                ]
            formatted_rows.append("[" + ", ".join(entries) + "]")
        return "[" + ", ".join(formatted_rows) + "]"

    @staticmethod
    def _format_numeric_entry(entry: object, precision: int) -> str:
        value = float(entry)
        rounded = round(value, precision)
        if abs(rounded) < 10 ** (-precision):
            rounded = 0.0
        text = f"{rounded:.{precision}f}".rstrip("0").rstrip(".")
        if text in {"", "-0"}:
            return "0"
        return text

    @staticmethod
    def _shannon_entropy(probabilities: np.ndarray, atol: float = 1e-9) -> float:
        """Return Shannon entropy for a numeric probability vector/table."""
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

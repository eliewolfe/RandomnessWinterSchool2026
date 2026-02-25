"""Contextuality scenario container for Bob-outcome behaviors.

Data conventions used in this module:
- Internally ``data`` has padded shape ``(X, Y, B_max)`` and stores
  ``p(b|x,y)``.
- Per-measurement valid outcomes are tracked by ``b_cardinality_per_y``.
- A single preparation operational equivalence has shape ``(X,)``.
- A list of preparation operational equivalences has shape ``(N_prep, X)``.
- A single measurement operational equivalence has shape ``(Y, B_max)``.
- A list of measurement operational equivalences has shape ``(N_meas, Y, B_max)``.
"""

from __future__ import annotations

import warnings
from functools import cached_property
from typing import Literal

from methodtools import lru_cache
import numpy as np
import sympy as sp

from ._ragged import (
    normalize_behavior_table_bob,
    normalize_opeq_array,
    structural_zero_opeqs,
)
from .linalg_utils import null_space_basis


class ContextualityScenario:
    """Container for Bob-outcome contextuality models used across this package."""

    data_symbolic: np.ndarray
    opeq_preps_symbolic: np.ndarray
    opeq_meas_symbolic: np.ndarray
    X_cardinality: int
    Y_cardinality: int
    B_cardinality: int
    _b_cardinality_per_y: np.ndarray
    _valid_b_mask: np.ndarray
    atol: float
    verbose: bool

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

        data_padded, self._b_cardinality_per_y, self._valid_b_mask = normalize_behavior_table_bob(
            data,
            atol=self.atol,
            pad_value=0,
        )
        self.data_symbolic = self._to_symbolic_array(data_padded)
        self.X_cardinality, self.Y_cardinality, self.B_cardinality = self.data_symbolic.shape

        if opeq_preps is None:
            if self.verbose:
                self._warn_verbose(
                    "Preparation operational equivalences were not provided; "
                    "discovering them from the data-table nullspace."
                )
            prep_base = self._to_symbolic_array(self.discover_opeqs_multisource())
        else:
            prep_base = self._to_symbolic_array(self._normalize_prep_opeq_array(opeq_preps))
            self.validate_opeqs_multisource(prep_base)

        if opeq_meas is None:
            if self.verbose:
                self._warn_verbose(
                    "Measurement operational equivalences were not provided; "
                    "discovering them from the data-table nullspace."
                )
            meas_base = self._to_symbolic_array(self.discover_opeqs_multimeter())
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
        self.opeq_preps_symbolic = prep_base
        self.opeq_meas_symbolic = self._append_structural_zero_opeqs(meas_base, meas_structural_zero)

        self.validate_opeqs_multisource(self.opeq_preps_symbolic)
        self.validate_opeqs_multimeter(self.opeq_meas_symbolic)

        if self.verbose:
            self._print_verbose_report()

    def __repr__(self) -> str:
        return (
            "ContextualityScenario("
            f"X={self.X_cardinality}, Y={self.Y_cardinality}, B={self.B_cardinality}, "
            f"B_per_y={self._b_cardinality_per_y.tolist()}, "
            f"num_opeq_preps={self.opeq_preps_symbolic.shape[0]}, "
            f"num_opeq_meas={self.opeq_meas_symbolic.shape[0]})"
        )

    @property
    def b_cardinality_per_y(self) -> np.ndarray:
        """Per-setting Bob outcome counts, shape ``(Y,)``."""
        return self._b_cardinality_per_y.copy()

    @property
    def valid_b_mask(self) -> np.ndarray:
        """Boolean mask for valid Bob outcomes, shape ``(Y, B)``."""
        return self._valid_b_mask.copy()

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
    def _normalize_contextuality_metric_name(metric: str) -> str:
        """Normalize contextuality metric labels used by print helpers."""
        token = str(metric).strip().lower().replace(" ", "_").replace("-", "_")
        token = "_".join(part for part in token.split("_") if part)
        aliases = {
            "dephasing_robustness": "dephasing_robustness",
            "robustness_to_dephasing": "dephasing_robustness",
            "contextual_fraction": "contextual_fraction",
            "noncontextual_fraction": "noncontextual_fraction",
        }
        if token not in aliases:
            raise ValueError(f"Unknown contextuality metric: {metric}")
        return aliases[token]

    @staticmethod
    def print_title(title: str, width: int = 80) -> None:
        """Print a section title with separator bars."""
        width_int = int(width)
        if width_int <= 0:
            raise ValueError("width must be positive.")
        print("\n" + "=" * width_int)
        print(str(title))
        print("=" * width_int)

    @staticmethod
    def format_numeric(value: object, precision: int = 3) -> str:
        """Format one numeric entry using scenario-consistent rounding rules."""
        return ContextualityScenario._format_numeric_entry(value, precision=int(precision))

    @staticmethod
    def binary_entropy(probability: float, atol: float = 1e-12) -> float:
        """Return binary Shannon entropy ``h2(p)``."""
        from .randomness import binary_entropy as _binary_entropy

        return float(_binary_entropy(float(probability), atol=float(atol)))

    @lru_cache(maxsize=None)
    def _compute_dephasing_robustness_cached(self, atol_value: float | None) -> float:
        """Cached dephasing robustness with default dephasing target."""
        from .contextuality import contextuality_robustness_to_dephasing

        return float(contextuality_robustness_to_dephasing(self, dephasing_target=None, atol=atol_value))

    def compute_dephasing_robustness(
        self,
        dephasing_target: np.ndarray | None = None,
        atol: float | None = None,
    ) -> float:
        """Compute contextuality robustness to dephasing."""
        if dephasing_target is None:
            atol_key = None if atol is None else float(atol)
            return float(self._compute_dephasing_robustness_cached(atol_key))

        from .contextuality import contextuality_robustness_to_dephasing

        return float(contextuality_robustness_to_dephasing(self, dephasing_target=dephasing_target, atol=atol))

    @lru_cache(maxsize=None)
    def _compute_contextual_fraction_cached(self, atol_value: float | None) -> float:
        """Cached contextual fraction for the scenario."""
        from .contextuality import contextual_fraction

        return float(contextual_fraction(self, atol=atol_value))

    def compute_contextual_fraction(self, atol: float | None = None) -> float:
        """Compute contextual fraction."""
        atol_key = None if atol is None else float(atol)
        return float(self._compute_contextual_fraction_cached(atol_key))

    @lru_cache(maxsize=None)
    def _compute_noncontextual_fraction_cached(self, atol_value: float | None) -> float:
        """Cached noncontextual fraction for the scenario."""
        from .contextuality import noncontextual_fraction

        return float(noncontextual_fraction(self, atol=atol_value))

    def compute_noncontextual_fraction(self, atol: float | None = None) -> float:
        """Compute noncontextual fraction."""
        atol_key = None if atol is None else float(atol)
        return float(self._compute_noncontextual_fraction_cached(atol_key))

    def print_preparation_index_sets(
        self,
        preparation_indices: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
    ) -> None:
        """Print preparation index sets used for grouped demos."""
        print("\nProvided preparation index sets:")
        for x, idx in enumerate(preparation_indices):
            print(f"x={x}: preparations {tuple(idx)}")

    def print_measurement_index_sets(
        self,
        measurement_indices: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
    ) -> None:
        """Print measurement index sets used for grouped demos."""
        print("\nProvided measurement index sets:")
        for y, idx in enumerate(measurement_indices):
            print(f"y={y}: effects {tuple(idx)}")

    def print_contextuality_measures(
        self,
        metrics: list[str] | tuple[str, ...] | None = None,
        *,
        precision: int = 3,
    ) -> None:
        """Print selected contextuality measures."""
        metric_list = ["dephasing_robustness", "contextual_fraction"] if metrics is None else list(metrics)
        canonical_metrics = [self._normalize_contextuality_metric_name(metric) for metric in metric_list]
        print("\nMeasures of Contextuality (closer to 1 means more contextual):")
        for metric in canonical_metrics:
            if metric == "dephasing_robustness":
                value = self.compute_dephasing_robustness()
                print(f"dephasing robustness = {self._format_numeric_entry(value, precision=precision)}")
            elif metric == "contextual_fraction":
                value = self.compute_contextual_fraction()
                print(f"contextual fraction = {self._format_numeric_entry(value, precision=precision)}")
            elif metric == "noncontextual_fraction":
                value = self.compute_noncontextual_fraction()
                print(f"noncontextual fraction = {self._format_numeric_entry(value, precision=precision)}")

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

    def discover_opeqs_multisource(self) -> np.ndarray:
        """Discover preparation OPEQs ``sum_x c[x] p(b|x,y)=0`` from data nullspace."""
        matrix = self.data_numeric.transpose(1, 2, 0).reshape(
            self.Y_cardinality * self.B_cardinality,
            self.X_cardinality,
        )
        basis = null_space_basis(matrix, atol=self.atol)
        return basis.reshape(-1, self.X_cardinality)

    def discover_opeqs_multimeter(self) -> np.ndarray:
        """Discover measurement OPEQs ``sum_{y,b} d[y,b] p(b|x,y)=0`` from data nullspace."""
        matrix = self.data_numeric.reshape(self.X_cardinality, self.Y_cardinality * self.B_cardinality)
        basis = null_space_basis(matrix, atol=self.atol)
        return basis.reshape(-1, self.Y_cardinality, self.B_cardinality)

    def validate_opeqs_multisource(self, opeqs: np.ndarray) -> None:
        """Validate preparation operational equivalences against data."""
        coeffs = self._to_float_array(self._normalize_prep_opeq_array(opeqs), atol=self.atol)
        residual = np.tensordot(coeffs, self.data_numeric, axes=([1], [0]))
        if not np.allclose(residual, 0.0, atol=self.atol):
            raise ValueError("Provided preparation operational equivalences are inconsistent with data.")

    def validate_opeqs_multimeter(self, opeqs: np.ndarray) -> None:
        """Validate measurement operational equivalences against data."""
        coeffs = self._to_float_array(
            normalize_opeq_array(
                opeqs,
                num_settings=self.Y_cardinality,
                outcome_cardinality_per_setting=self._b_cardinality_per_y,
                max_outcomes=self.B_cardinality,
                name="Measurement operational equivalences",
                pad_value=0,
            ),
            atol=self.atol,
        )
        residual = np.tensordot(coeffs, self.data_numeric, axes=([1, 2], [1, 2]))
        if not np.allclose(residual, 0.0, atol=self.atol):
            raise ValueError("Provided measurement operational equivalences are inconsistent with data.")

    def sanity_check(self) -> None:
        """Run basic consistency checks for data normalization and opeqs."""
        numeric_data = self.data_numeric
        if numeric_data.ndim != 3:
            raise ValueError("data must have exactly 3 indices.")
        if np.any(numeric_data < -self.atol):
            raise ValueError("data contains negative probabilities.")

        invalid = np.broadcast_to(~self._valid_b_mask[np.newaxis, :, :], numeric_data.shape)
        invalid_entries = np.abs(numeric_data[invalid])
        if invalid_entries.size and np.any(invalid_entries > self.atol):
            raise ValueError("data has nonzero entries in padded (invalid) coordinates.")

        if not np.allclose(numeric_data.sum(axis=2), 1.0, atol=self.atol):
            raise ValueError("Each (x, y) slice must satisfy sum_b p(b|x,y) = 1.")
        self.validate_opeqs_multisource(self.opeq_preps_symbolic)
        self.validate_opeqs_multimeter(self.opeq_meas_symbolic)

    def format_probabilities(
        self,
        as_p_b_given_x_y: bool = True,
        precision: int = 6,
        representation: Literal["numeric", "symbolic"] = "numeric",
    ) -> str:
        """Return a readable probability-table string."""
        data = self._array_for_representation(
            representation=representation,
            numeric=self.data_numeric,
            symbolic=self.data_symbolic,
        )
        lines = []
        if as_p_b_given_x_y:
            for x in range(self.X_cardinality):
                matrix = data[x, :, :]
                lines.append(f"x={x}")
                lines.append(self._format_matrix(matrix, precision=precision, representation=representation))
            return "\n".join(lines)

        for x in range(self.X_cardinality):
            for y in range(self.Y_cardinality):
                b_count = int(self._b_cardinality_per_y[y])
                row = data[x, y, :b_count]
                lines.append(f"x={x}, y={y}")
                lines.append(self._format_matrix(row[np.newaxis, :], precision=precision, representation=representation))
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
        prep_keep = self._nontrivial_prep_opeq_indices(self.opeq_preps_numeric)
        for k_display, k_source in enumerate(prep_keep):
            row = np.asarray(prep_eqs_repr[k_source], dtype=object).reshape(1, -1)
            eq_line = self._format_ragged_matrix_single_line(
                rows=[[entry for entry in row[0].tolist()]],
                precision=precision,
                representation=representation,
            )
            lines.append(f"k={k_display}: {eq_line}")

        lines.append("Measurement OPEQs:")
        meas_keep = self._nontrivial_meas_opeq_indices(self.opeq_meas_numeric)
        for k_display, k_source in enumerate(meas_keep):
            eq_rows = self._trim_meas_opeq_rows(meas_eqs_repr[k_source])
            eq_line = self._format_ragged_matrix_single_line(
                eq_rows,
                precision=precision,
                representation=representation,
            )
            lines.append(f"k={k_display}: {eq_line}")
        return "\n".join(lines)

    def print_probabilities(
        self,
        as_p_b_given_x_y: bool = True,
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
        meas_keep = self._nontrivial_meas_opeq_indices(self.opeq_meas_numeric)
        for k_display, k_source in enumerate(meas_keep):
            eq_rows = self._trim_meas_opeq_rows(meas_eqs_repr[k_source])
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
        print("\nData table p(b|x,y):")
        self.print_probabilities(as_p_b_given_x_y=True, representation=representation)
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
        """Append padded-coordinate zero constraints to a measurement OPEQ list."""
        base = np.asarray(base_opeqs, dtype=object)
        if base.ndim == 2:
            base = base[np.newaxis, :, :]
        if base.ndim != 3:
            raise ValueError("base_opeqs must have 2D or 3D shape.")

        zeros = np.asarray(structural_zeros, dtype=object)
        if zeros.ndim == 2:
            zeros = zeros[np.newaxis, :, :]
        if zeros.ndim != 3:
            raise ValueError("structural_zeros must have 2D or 3D shape.")
        if zeros.shape[0] == 0:
            return base
        if base.shape[1:] != zeros.shape[1:]:
            raise ValueError("base_opeqs and structural_zeros must share setting/outcome dimensions.")
        return np.concatenate([base, zeros], axis=0)

    def _normalize_prep_opeq_array(self, values: object) -> np.ndarray:
        arr = np.asarray(values, dtype=object)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2:
            raise ValueError("Preparation operational equivalences must have shape (N_prep, X) or (X,).")
        if arr.shape[1] != self.X_cardinality:
            raise ValueError(
                "Preparation operational equivalences have incompatible X dimension: "
                f"got {arr.shape[1]}, expected {self.X_cardinality}."
            )
        return arr

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

    def _nontrivial_prep_opeq_indices(self, opeqs_numeric: np.ndarray) -> list[int]:
        arr = np.asarray(opeqs_numeric, dtype=float)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2:
            raise ValueError("Preparation OPEQ array must be 1D or 2D.")

        keep: list[int] = []
        for k in range(arr.shape[0]):
            coeffs = arr[k]
            has_positive = bool(np.any(coeffs > self.atol))
            has_negative = bool(np.any(coeffs < -self.atol))
            if has_positive and has_negative:
                keep.append(k)
        return keep

    def _nontrivial_meas_opeq_indices(self, opeqs_numeric: np.ndarray) -> list[int]:
        arr = np.asarray(opeqs_numeric, dtype=float)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        if arr.ndim != 3:
            raise ValueError("Measurement OPEQ array must have 2D or 3D shape.")

        keep: list[int] = []
        for k in range(arr.shape[0]):
            rows = self._trim_meas_opeq_rows(arr[k])
            coeffs = np.asarray([entry for row in rows for entry in row], dtype=float)
            if coeffs.size == 0:
                continue
            has_positive = bool(np.any(coeffs > self.atol))
            has_negative = bool(np.any(coeffs < -self.atol))
            if has_positive and has_negative:
                keep.append(k)
        return keep

    def _trim_meas_opeq_rows(self, eq_matrix: np.ndarray) -> list[list[object]]:
        eq = np.asarray(eq_matrix, dtype=object)
        if eq.ndim != 2:
            raise ValueError("Measurement OPEQ matrix must be 2D.")
        if eq.shape[0] != self.Y_cardinality:
            raise ValueError("Measurement OPEQ matrix has incompatible setting dimension.")

        out: list[list[object]] = []
        for y in range(self.Y_cardinality):
            count = int(self._b_cardinality_per_y[y])
            out.append([eq[y, b] for b in range(count)])
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
        return "[" + ", ".join(formatted_rows) + "]" if formatted_rows else "[]"

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

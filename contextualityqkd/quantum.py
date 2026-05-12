"""Quantum and GPT utilities for contextuality workflows."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from typing import Sequence

import numpy as np
import sympy as sp

from ._ragged import (
    embed_basis_rows_to_padded,
    flatten_valid_indices,
    normalize_grouped_vectors_settings_single_outcome,
    normalize_grouped_vectors_single_setting_many_outcomes,
)
from .linalg_utils import null_space_basis
from .scenario import ContextualityScenario


class GPTContextualityScenario(ContextualityScenario):
    """Contextuality scenario built directly from GPT primitives with one state per x."""

    gpt_states_grouped: np.ndarray
    gpt_effects_grouped: np.ndarray
    measurement_indices: tuple[tuple[int, ...], ...]

    def __init__(
        self,
        gpt_states: object,
        gpt_effects: object,
        *,
        measurement_indices: Sequence[Sequence[int]] | None = None,
        infer_measurement_indices: bool = False,
        unit_effect: np.ndarray | None = None,
        atol: float = 1e-9,
        outcomes_per_measurement: int | None = None,
        drop_tiny_imag: bool = True,
        verbose: bool = False,
    ) -> None:
        if gpt_states is None or gpt_effects is None:
            raise ValueError("gpt_states and gpt_effects are required.")
        if measurement_indices is not None and infer_measurement_indices:
            raise ValueError(
                "Provide measurement_indices or set infer_measurement_indices=True, not both."
            )
        self.atol = float(atol)
        self._input_gpt_states = deepcopy(gpt_states)
        self._input_gpt_effects = deepcopy(gpt_effects)
        self._input_measurement_indices = (
            None
            if measurement_indices is None
            else tuple(tuple(int(entry) for entry in group) for group in measurement_indices)
        )
        self._input_infer_measurement_indices = bool(infer_measurement_indices)
        self._input_unit_effect = None if unit_effect is None else np.asarray(unit_effect, dtype=object).copy()
        if self._input_unit_effect is not None:
            self._input_unit_effect.setflags(write=False)
        self._input_outcomes_per_measurement = outcomes_per_measurement
        self._input_drop_tiny_imag = bool(drop_tiny_imag)
        self._measurement_indices_override: tuple[tuple[int, ...], ...] | None = None

        states_flat, states_grouped, effects_grouped, measurement_indices_resolved = self._resolve_gpt_inputs()

        states_dense, _state_counts, _ = normalize_grouped_vectors_settings_single_outcome(
            states_grouped,
            name="gpt_states",
        )
        effects_dense, b_cardinality_per_y, _ = normalize_grouped_vectors_settings_single_outcome(
            effects_grouped,
            name="gpt_effects",
        )

        data_table_4d = self.probability_table_from_gpt_vectors(
            gpt_states=states_grouped,
            gpt_effects=effects_grouped,
            normalize_source_outcomes=True,
            atol=float(atol),
            drop_tiny_imag=self._input_drop_tiny_imag,
        )
        data_dense_4d = np.asarray(np.ma.getdata(data_table_4d), dtype=object).copy()
        if np.ma.isMaskedArray(data_table_4d):
            data_dense_4d[np.asarray(np.ma.getmaskarray(data_table_4d), dtype=bool)] = 0
        data_table = data_dense_4d[:, :, 0, :]

        prep_opeqs_raw = self.discover_operational_equivalences_from_gpt_objects(
            states_flat[:, np.newaxis, :],
            atol=float(atol),
        )
        opeq_preps = np.asarray(prep_opeqs_raw, dtype=object).reshape(-1, states_flat.shape[0])

        effects_for_opeq = [
            [effects_dense[y, b, :] for b in range(int(b_cardinality_per_y[y]))]
            for y in range(effects_dense.shape[0])
        ]
        opeq_meas = self.discover_operational_equivalences_from_gpt_objects(
            effects_for_opeq,
            atol=float(atol),
        )

        states_grouped_trimmed = np.array(
            [[states_dense[x, 0, :]] for x in range(states_dense.shape[0])],
            dtype=object,
        )
        effects_grouped_trimmed = self._grouped_vectors_from_dense(effects_dense, b_cardinality_per_y)

        super().__init__(
            data=self._freeze_object_array(data_table),
            opeq_preps=self._freeze_object_array(opeq_preps),
            opeq_meas=self._freeze_object_array(opeq_meas),
            atol=float(atol),
            verbose=verbose,
        )

        self._gpt_states_grouped_data = self._freeze_object_array(states_grouped_trimmed)
        self._gpt_effects_grouped_data = self._freeze_object_array(effects_grouped_trimmed)
        self._measurement_indices_data = tuple(
            tuple(int(entry) for entry in combo) for combo in measurement_indices_resolved
        )

        if verbose and infer_measurement_indices:
            print("\nInferred measurement index sets:")
            for y, idx in enumerate(self.measurement_indices):
                print(f"y={y}: effects {idx}")

    @staticmethod
    def maximally_mixed_vector(d: int) -> np.ndarray:
        """Return GPT vector for maximally mixed state ``I/d`` in this basis."""
        vec = np.zeros(d * d, dtype=float)
        vec[0] = 1.0 / np.sqrt(d)
        return vec

    @staticmethod
    def unit_effect_vector(d: int) -> np.ndarray:
        """Return GPT vector for unit effect ``I`` in this basis."""
        vec = np.zeros(d * d, dtype=float)
        vec[0] = np.sqrt(d)
        return vec

    @staticmethod
    def xz_plane_ket(theta: object) -> object:
        """Return real-amplitude qubit ket in the X-Z Bloch plane."""
        theta_sym = sp.sympify(theta)
        return sp.Matrix([sp.cos(theta_sym / 2), sp.sin(theta_sym / 2)])

    @staticmethod
    def normalize_integer_rays_symbolic(rays: np.ndarray) -> list[object]:
        """Normalize integer-valued ray rows exactly using SymPy."""
        ray_array = np.asarray(rays, dtype=int)
        if ray_array.ndim != 2:
            raise ValueError("rays must have shape (N, d).")

        normalized_kets: list[object] = []
        for row in ray_array:
            row_sym = [sp.Integer(int(entry)) for entry in row]
            norm_sq = sum(entry * entry for entry in row_sym)
            if norm_sq == 0:
                raise ValueError("Cannot normalize zero ray.")
            norm = sp.sqrt(norm_sq)
            normalized_kets.append(sp.Matrix([entry / norm for entry in row_sym]))
        return normalized_kets

    @staticmethod
    def group_gpt_vectors_by_indices(
        gpt_vector_set: np.ndarray,
        grouped_indices: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
    ) -> np.ndarray:
        """Group flat GPT vectors by setting-index tuples into ragged-friendly arrays."""
        return np.array(
            [[gpt_vector_set[idx] for idx in index_group] for index_group in grouped_indices],
            dtype=object,
        )

    @classmethod
    def from_xz_ring(
        cls,
        *,
        num_states: int,
        measurement_indices: Sequence[Sequence[int]] | None = None,
        infer_measurement_indices: bool = False,
        unit_effect: np.ndarray | None = None,
        atol: float = 1e-9,
        outcomes_per_measurement: int | None = None,
        drop_tiny_imag: bool = True,
        verbose: bool = False,
    ) -> "GPTContextualityScenario":
        """Build a GPT scenario from uniformly spaced XZ-ring states/effects."""
        num_states_int = int(num_states)
        if num_states_int < 4:
            raise ValueError("num_states must be at least 4.")
        if num_states_int % 2 != 0:
            raise ValueError("num_states must be even for antipodal ring measurements.")

        thetas = [sp.Integer(k) * 2 * sp.pi / num_states_int for k in range(num_states_int)]
        kets = [cls.xz_plane_ket(theta) for theta in thetas]
        gpt_set = np.array([cls.projector_hs_vector(ket, drop_tiny_imag=drop_tiny_imag) for ket in kets], dtype=object)

        resolved_measurements = measurement_indices
        if resolved_measurements is None and not infer_measurement_indices:
            resolved_measurements = tuple((i, i + num_states_int // 2) for i in range(num_states_int // 2))

        return cls(
            gpt_states=gpt_set,
            gpt_effects=gpt_set,
            measurement_indices=resolved_measurements,
            infer_measurement_indices=infer_measurement_indices,
            unit_effect=unit_effect,
            atol=atol,
            outcomes_per_measurement=outcomes_per_measurement,
            drop_tiny_imag=drop_tiny_imag,
            verbose=verbose,
        )

    @classmethod
    def from_integer_rays(
        cls,
        *,
        rays: np.ndarray,
        measurement_indices: Sequence[Sequence[int]] | None = None,
        infer_measurement_indices: bool = False,
        unit_effect: np.ndarray | None = None,
        atol: float = 1e-9,
        outcomes_per_measurement: int | None = None,
        drop_tiny_imag: bool = True,
        verbose: bool = False,
    ) -> "GPTContextualityScenario":
        """Build a GPT scenario from integer rays via symbolic normalization and HS projection."""
        kets = cls.normalize_integer_rays_symbolic(rays)
        gpt_set = np.array([cls.projector_hs_vector(ket, drop_tiny_imag=drop_tiny_imag) for ket in kets], dtype=object)
        return cls(
            gpt_states=gpt_set,
            gpt_effects=gpt_set,
            measurement_indices=measurement_indices,
            infer_measurement_indices=infer_measurement_indices,
            unit_effect=unit_effect,
            atol=atol,
            outcomes_per_measurement=outcomes_per_measurement,
            drop_tiny_imag=drop_tiny_imag,
            verbose=verbose,
        )

    @classmethod
    def projector_hs_vector(cls, ket: object, drop_tiny_imag: bool = True) -> np.ndarray:
        """Return Hilbert-Schmidt vectorization of ``|psi><psi|``."""
        if cls._does_not_contain_floats(ket):
            proj_sym = cls._projector_from_ket_sympy(ket, drop_tiny_imag=False)
            return cls._matrix_to_hs_vector_sympy(proj_sym, drop_tiny_imag=drop_tiny_imag)
        proj_num = cls._projector_from_ket_numpy(ket, drop_tiny_imag=False)
        return cls._matrix_to_hs_vector(proj_num, drop_tiny_imag=drop_tiny_imag)

    @classmethod
    def _does_not_contain_floats(cls, obj: object) -> bool:
        """Return True when no machine float/complex entries are present."""
        if isinstance(obj, (float, complex, np.floating, np.complexfloating)):
            return False
        if isinstance(obj, sp.MatrixBase):
            return all(cls._does_not_contain_floats(entry) for entry in obj)
        if isinstance(obj, np.ndarray):
            if np.issubdtype(obj.dtype, np.floating) or np.issubdtype(obj.dtype, np.complexfloating):
                return False
            if obj.dtype != object:
                return True
            return all(cls._does_not_contain_floats(entry) for entry in obj.reshape(-1))
        if isinstance(obj, (list, tuple)):
            return all(cls._does_not_contain_floats(entry) for entry in obj)
        return True

    @staticmethod
    def _to_sympy_object_array(values: object) -> np.ndarray:
        return np.asarray(values, dtype=object)

    @staticmethod
    def _sympy_numeric_complex(value: object) -> complex | None:
        try:
            return complex(sp.N(value))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _sympy_numeric_abs(cls, value: object) -> float | None:
        num = cls._sympy_numeric_complex(value)
        if num is None:
            return None
        return float(abs(num))

    @staticmethod
    def _assert_machine_numeric_real(values: object, *, name: str, atol: float) -> None:
        """Reject machine-numeric arrays with significant imaginary components."""
        try:
            arr = np.asarray(values, dtype=complex)
        except (TypeError, ValueError):
            return
        if arr.size and np.max(np.abs(np.imag(arr))) > float(atol):
            raise ValueError(f"{name} contains significant imaginary components.")

    @classmethod
    def _coerce_distribution_rows(
        cls,
        source_outcome_distribution: object,
        *,
        num_x: int,
    ) -> list[list[object]]:
        arr = np.asarray(source_outcome_distribution, dtype=object)
        if arr.ndim == 2:
            if arr.shape[0] != num_x:
                raise ValueError(f"source_outcome_distribution must have {num_x} rows.")
            return [arr[x, :].tolist() for x in range(num_x)]

        if not isinstance(source_outcome_distribution, (list, tuple, np.ndarray)):
            raise ValueError("source_outcome_distribution must be a 2D array or nested list-like input.")
        rows = list(source_outcome_distribution)
        if len(rows) != num_x:
            raise ValueError(f"source_outcome_distribution must have {num_x} rows.")

        out: list[list[object]] = []
        for x, row in enumerate(rows):
            if isinstance(row, np.ndarray):
                out.append(np.asarray(row, dtype=object).reshape(-1).tolist())
                continue
            if isinstance(row, (list, tuple)):
                out.append(list(row))
                continue
            raise ValueError(f"source_outcome_distribution row {x} is not list-like.")
        return out

    @classmethod
    def _normalize_source_outcome_distribution_numeric(
        cls,
        source_outcome_distribution: object | None,
        *,
        a_cardinality_per_x: np.ndarray,
        num_x: int,
        a_max: int,
        atol: float,
    ) -> np.ndarray:
        counts = np.asarray(a_cardinality_per_x, dtype=int).reshape(-1)
        if counts.shape != (num_x,):
            raise ValueError(f"a_cardinality_per_x must have shape ({num_x},).")

        p_a_given_x = np.zeros((num_x, a_max), dtype=float)
        if source_outcome_distribution is None:
            for x in range(num_x):
                count = int(counts[x])
                p_a_given_x[x, :count] = 1.0 / float(count)
            return p_a_given_x

        rows = cls._coerce_distribution_rows(source_outcome_distribution, num_x=num_x)
        for x in range(num_x):
            row = rows[x]
            count = int(counts[x])
            if len(row) not in {count, a_max}:
                raise ValueError(
                    f"source_outcome_distribution row {x} must have length {count} "
                    f"(or padded length {a_max}), got {len(row)}."
                )
            row_arr = np.asarray(row, dtype=complex)
            if row_arr.size and np.max(np.abs(np.imag(row_arr))) > float(atol):
                raise ValueError("source_outcome_distribution contains significant imaginary entries.")
            row_real = np.asarray(np.real(row_arr), dtype=float)

            if len(row) == a_max and count < a_max:
                padded_values = row_real[count:]
                if padded_values.size and np.any(np.abs(padded_values) > float(atol)):
                    raise ValueError("source_outcome_distribution has nonzero padded entries.")

            valid_values = row_real[:count]
            if np.any(valid_values < -float(atol)):
                raise ValueError("source_outcome_distribution contains negative entries.")
            if not np.allclose(np.sum(valid_values), 1.0, atol=float(atol)):
                raise ValueError("Each source_outcome_distribution[x,:] must sum to 1 on valid outcomes.")
            p_a_given_x[x, :count] = np.clip(valid_values, 0.0, None)
        return p_a_given_x

    @classmethod
    def _normalize_source_outcome_distribution_symbolic(
        cls,
        source_outcome_distribution: object | None,
        *,
        a_cardinality_per_x: np.ndarray,
        num_x: int,
        a_max: int,
        atol: float,
    ) -> np.ndarray:
        counts = np.asarray(a_cardinality_per_x, dtype=int).reshape(-1)
        if counts.shape != (num_x,):
            raise ValueError(f"a_cardinality_per_x must have shape ({num_x},).")

        p_a_given_x = np.empty((num_x, a_max), dtype=object)
        p_a_given_x[:, :] = sp.Integer(0)

        if source_outcome_distribution is None:
            for x in range(num_x):
                count = int(counts[x])
                uniform = sp.Rational(1, count)
                for a in range(count):
                    p_a_given_x[x, a] = uniform
            return p_a_given_x

        rows = cls._coerce_distribution_rows(source_outcome_distribution, num_x=num_x)
        for x in range(num_x):
            row = rows[x]
            count = int(counts[x])
            if len(row) not in {count, a_max}:
                raise ValueError(
                    f"source_outcome_distribution row {x} must have length {count} "
                    f"(or padded length {a_max}), got {len(row)}."
                )

            row_sum = sp.Integer(0)
            for a in range(count):
                value = sp.sympify(row[a])
                imag_part = sp.im(value)
                imag_abs = cls._sympy_numeric_abs(imag_part)
                if imag_part.is_zero is not True and (imag_abs is None or imag_abs > float(atol)):
                    raise ValueError("source_outcome_distribution contains significant imaginary entries.")
                numeric_value = cls._sympy_numeric_complex(value)
                if numeric_value is not None and numeric_value.real < -float(atol):
                    raise ValueError("source_outcome_distribution contains negative entries.")
                p_a_given_x[x, a] = value
                row_sum += value

            if len(row) == a_max and count < a_max:
                for a in range(count, a_max):
                    padded_value = sp.sympify(row[a])
                    padded_abs = cls._sympy_numeric_abs(padded_value)
                    if padded_abs is not None:
                        if padded_abs > float(atol):
                            raise ValueError("source_outcome_distribution has nonzero padded entries.")
                    elif padded_value.is_zero is not True:
                        raise ValueError("source_outcome_distribution padded entries must be zero.")

            row_gap = row_sum - 1
            row_gap_abs = cls._sympy_numeric_abs(row_gap)
            if row_gap.is_zero is not True and (row_gap_abs is None or row_gap_abs > float(atol)):
                raise ValueError("Each source_outcome_distribution[x,:] must sum to 1 on valid outcomes.")

        return p_a_given_x

    @staticmethod
    def _projector_from_ket_numpy(ket: object, drop_tiny_imag: bool = True) -> np.ndarray:
        vec = np.asarray(ket, dtype=complex)
        if vec.ndim != 1:
            raise ValueError("ket must be a 1D state vector.")
        proj = np.outer(vec, vec.conj())
        return np.real_if_close(proj) if drop_tiny_imag else proj

    @staticmethod
    def _as_sympy_column_vector(ket: object) -> object:
        if isinstance(ket, sp.MatrixBase):
            vec = ket
        else:
            arr = np.asarray(ket, dtype=object)
            if arr.ndim == 2 and 1 in arr.shape:
                arr = arr.reshape(-1)
            if arr.ndim != 1:
                raise ValueError("ket must be a 1D state vector.")
            vec = sp.Matrix(arr.tolist())

        if vec.cols == 1:
            return vec
        if vec.rows == 1:
            return vec.T
        raise ValueError("ket must be a 1D state vector.")

    @classmethod
    def _projector_from_ket_sympy(cls, ket: object, drop_tiny_imag: bool = True) -> np.ndarray:
        vec = cls._as_sympy_column_vector(ket)
        proj = vec * vec.conjugate().T
        out = np.empty((proj.rows, proj.cols), dtype=object)
        for i in range(proj.rows):
            for j in range(proj.cols):
                entry = proj[i, j]
                if drop_tiny_imag and entry.is_real is True:
                    entry = sp.re(entry)
                out[i, j] = entry
        return out

    @staticmethod
    def _matrix_to_hs_vector(matrix: np.ndarray, drop_tiny_imag: bool = True) -> np.ndarray:
        """Vectorize one matrix as ``vec(M^T)`` for trace-compatible dot products."""
        mat = np.asarray(matrix, dtype=complex)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError("matrix must have shape (d, d).")
        vec = np.swapaxes(mat, -2, -1).reshape(-1)
        return np.real_if_close(vec) if drop_tiny_imag else vec

    @staticmethod
    def _matrix_to_hs_vector_sympy(matrix: object, drop_tiny_imag: bool = True) -> np.ndarray:
        if isinstance(matrix, sp.MatrixBase):
            mat = matrix
        else:
            arr = np.asarray(matrix, dtype=object)
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                raise ValueError("matrix must have shape (d, d).")
            mat = sp.Matrix(arr.tolist())

        if mat.rows != mat.cols:
            raise ValueError("matrix must have shape (d, d).")

        vec: list[object] = []
        for i in range(mat.rows):
            for j in range(mat.cols):
                entry = mat[j, i]
                if drop_tiny_imag and entry.is_real is True:
                    entry = sp.re(entry)
                vec.append(entry)
        return np.asarray(vec, dtype=object)

    @staticmethod
    def _coerce_flat_vector_set(values: object, *, name: str) -> np.ndarray:
        """Return flat GPT vectors with shape ``(N, K)``."""
        flat = np.asarray(values, dtype=object)
        if flat.ndim != 2:
            raise ValueError(f"{name} must have flat shape (N, K) when explicit indices/inference are used.")
        if flat.shape[0] == 0 or flat.shape[1] == 0:
            raise ValueError(f"{name} must be non-empty.")
        return flat

    @staticmethod
    def _normalize_index_groups(
        grouped_indices: Sequence[Sequence[int]],
        *,
        total_items: int,
        name: str,
    ) -> tuple[tuple[int, ...], ...]:
        """Validate and normalize grouping indices to nested tuples."""
        groups = tuple(tuple(int(entry) for entry in group) for group in grouped_indices)
        if not groups:
            raise ValueError(f"{name} must contain at least one setting.")
        for setting_idx, group in enumerate(groups):
            if not group:
                raise ValueError(f"{name}[{setting_idx}] must contain at least one index.")
            for idx in group:
                if idx < 0 or idx >= int(total_items):
                    raise ValueError(
                        f"{name}[{setting_idx}] contains out-of-range index {idx}; "
                        f"valid range is [0, {int(total_items) - 1}]."
                    )
        return groups

    @classmethod
    def _resolve_grouped_gpt_vectors(
        cls,
        gpt_values: object,
        *,
        grouped_indices: Sequence[Sequence[int]] | None,
        name: str,
    ) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
        """Resolve grouped GPT vectors and index metadata."""
        if grouped_indices is not None:
            flat = cls._coerce_flat_vector_set(gpt_values, name=name)
            normalized_groups = cls._normalize_index_groups(
                grouped_indices,
                total_items=flat.shape[0],
                name=f"{name}_indices",
            )
            grouped = cls.group_gpt_vectors_by_indices(flat, normalized_groups)
            return grouped, normalized_groups

        grouped, counts, _ = normalize_grouped_vectors_settings_single_outcome(
            gpt_values,
            name=name,
        )
        local_groups = tuple(tuple(range(int(count))) for count in counts.tolist())
        return grouped, local_groups

    @staticmethod
    def _grouped_vectors_from_dense(dense: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """Trim padded grouped vectors to ragged object-array form."""
        return np.array(
            [[dense[s, o, :] for o in range(int(counts[s]))] for s in range(dense.shape[0])],
            dtype=object,
        )

    @classmethod
    def _probability_table_from_gpt_vectors_symbolic(
        cls,
        gpt_states: np.ndarray,
        gpt_effects: np.ndarray,
        source_outcome_distribution: np.ndarray | None,
        normalize_source_outcomes: bool,
        return_masked: bool | None,
        atol: float,
        drop_tiny_imag: bool,
    ) -> np.ndarray | np.ma.MaskedArray:
        states_raw, a_cardinality_per_x, valid_a_mask = normalize_grouped_vectors_settings_single_outcome(
            gpt_states,
            name="gpt_states",
        )
        effects_raw, b_cardinality_per_y, valid_b_mask = normalize_grouped_vectors_settings_single_outcome(
            gpt_effects,
            name="gpt_effects",
        )
        states = cls._to_sympy_object_array(states_raw)
        effects = cls._to_sympy_object_array(effects_raw)
        if states.ndim != 3:
            raise ValueError("gpt_states must have shape (X, A, K) or ragged equivalent.")
        if effects.ndim != 3:
            raise ValueError("gpt_effects must have shape (Y, B, K) or ragged equivalent.")
        if states.shape[-1] != effects.shape[-1]:
            raise ValueError("State/effect vector dimensions do not match.")

        valid_ab_mask = valid_a_mask[:, np.newaxis, :, np.newaxis] & valid_b_mask[np.newaxis, :, np.newaxis, :]
        probs = np.einsum("xak,ybk->xyab", states, effects)
        if np.any(~valid_ab_mask):
            for idx in np.argwhere(~valid_ab_mask):
                probs[tuple(idx.tolist())] = sp.Integer(0)

        if normalize_source_outcomes:
            num_x, _, num_a, _ = probs.shape
            p_a_given_x = cls._normalize_source_outcome_distribution_symbolic(
                source_outcome_distribution=source_outcome_distribution,
                a_cardinality_per_x=a_cardinality_per_x,
                num_x=num_x,
                a_max=num_a,
                atol=atol,
            )
            probs = probs * p_a_given_x[:, np.newaxis, :, np.newaxis]

        if drop_tiny_imag:
            for value in np.asarray(probs, dtype=object).reshape(-1):
                imag_part = sp.im(value)
                imag_abs = cls._sympy_numeric_abs(imag_part)
                if imag_part.is_zero is not True and (imag_abs is None or imag_abs > float(atol)):
                    raise ValueError("Symbolic GPT probabilities contain significant imaginary components.")

        auto_masked = (
            not np.all(a_cardinality_per_x == a_cardinality_per_x[0])
            or not np.all(b_cardinality_per_y == b_cardinality_per_y[0])
        )
        use_masked = auto_masked if return_masked is None else bool(return_masked)
        if use_masked:
            return np.ma.array(probs, mask=~valid_ab_mask, copy=False)
        return probs

    @classmethod
    def probability_table_from_gpt_vectors(
        cls,
        gpt_states: np.ndarray,
        gpt_effects: np.ndarray,
        source_outcome_distribution: np.ndarray | None = None,
        normalize_source_outcomes: bool = True,
        return_masked: bool | None = None,
        atol: float = 1e-9,
        drop_tiny_imag: bool = True,
    ) -> np.ndarray | np.ma.MaskedArray:
        """Compute GPT probability table from grouped state/effect vectors."""
        if cls._does_not_contain_floats(gpt_states) and cls._does_not_contain_floats(gpt_effects):
            return cls._probability_table_from_gpt_vectors_symbolic(
                gpt_states=gpt_states,
                gpt_effects=gpt_effects,
                source_outcome_distribution=source_outcome_distribution,
                normalize_source_outcomes=normalize_source_outcomes,
                return_masked=return_masked,
                atol=atol,
                drop_tiny_imag=drop_tiny_imag,
            )

        states_raw, a_cardinality_per_x, valid_a_mask = normalize_grouped_vectors_settings_single_outcome(
            gpt_states,
            name="gpt_states",
        )
        effects_raw, b_cardinality_per_y, valid_b_mask = normalize_grouped_vectors_settings_single_outcome(
            gpt_effects,
            name="gpt_effects",
        )
        states = np.asarray(states_raw, dtype=complex)
        effects = np.asarray(effects_raw, dtype=complex)
        if states.ndim != 3:
            raise ValueError("gpt_states must have shape (X, A, K) or ragged equivalent.")
        if effects.ndim != 3:
            raise ValueError("gpt_effects must have shape (Y, B, K) or ragged equivalent.")
        if states.shape[-1] != effects.shape[-1]:
            raise ValueError("State/effect vector dimensions do not match.")

        valid_ab_mask = valid_a_mask[:, np.newaxis, :, np.newaxis] & valid_b_mask[np.newaxis, :, np.newaxis, :]
        probs = np.einsum("xak,ybk->xyab", states, effects)
        probs = np.where(valid_ab_mask, probs, 0.0)

        if normalize_source_outcomes:
            p_a_given_x = cls._normalize_source_outcome_distribution_numeric(
                source_outcome_distribution=source_outcome_distribution,
                a_cardinality_per_x=a_cardinality_per_x,
                num_x=states.shape[0],
                a_max=states.shape[1],
                atol=atol,
            )
            probs = probs * p_a_given_x[:, np.newaxis, :, np.newaxis]

        out = np.real_if_close(probs) if drop_tiny_imag else probs
        auto_masked = (
            not np.all(a_cardinality_per_x == a_cardinality_per_x[0])
            or not np.all(b_cardinality_per_y == b_cardinality_per_y[0])
        )
        use_masked = auto_masked if return_masked is None else bool(return_masked)
        if use_masked:
            return np.ma.array(out, mask=~valid_ab_mask, copy=False)
        return out

    @classmethod
    def _discover_operational_equivalences_from_gpt_objects_symbolic(
        cls,
        gpt_objects: np.ndarray,
        atol: float,
    ) -> np.ndarray:
        raw_grouped, _counts, valid_mask = normalize_grouped_vectors_single_setting_many_outcomes(
            gpt_objects,
            name="gpt_objects",
        )
        raw = cls._to_sympy_object_array(raw_grouped)
        if raw.ndim != 3:
            raise ValueError("gpt_objects must have shape (O,K)/(S,O,K) or ragged equivalent.")

        for value in np.asarray(raw, dtype=object).reshape(-1):
            imag_part = sp.im(value)
            imag_abs = cls._sympy_numeric_abs(imag_part)
            if imag_part.is_zero is not True and (imag_abs is None or imag_abs > float(atol)):
                raise ValueError("gpt_objects contains significant imaginary components.")

        num_s, num_o, vec_dim = raw.shape
        matrix_full = raw.reshape(num_s * num_o, vec_dim)
        valid_indices = flatten_valid_indices(valid_mask)
        matrix = matrix_full[valid_indices, :]
        sym_matrix = sp.Matrix(
            [[matrix[row, col] for col in range(matrix.shape[1])] for row in range(matrix.shape[0])]
        )
        nullspace_cols = sym_matrix.T.nullspace()
        if not nullspace_cols:
            basis_reduced = np.empty((0, matrix.shape[0]), dtype=object)
        else:
            basis_reduced = np.empty((len(nullspace_cols), matrix.shape[0]), dtype=object)
            for basis_idx, col in enumerate(nullspace_cols):
                for row_idx in range(matrix.shape[0]):
                    basis_reduced[basis_idx, row_idx] = col[row_idx]
        basis_full = embed_basis_rows_to_padded(
            basis_reduced,
            valid_flat_indices=valid_indices,
            total_size=num_s * num_o,
        )
        basis = basis_full.reshape(-1, num_s, num_o)
        if np.any(~valid_mask):
            for idx in np.argwhere(~valid_mask):
                basis[:, idx[0], idx[1]] = sp.Integer(0)
        return basis

    @classmethod
    def discover_operational_equivalences_from_gpt_objects(
        cls,
        gpt_objects: np.ndarray,
        atol: float = 1e-9,
    ) -> np.ndarray:
        """Discover operational equivalences from GPT vectors via nullspace."""
        if cls._does_not_contain_floats(gpt_objects):
            return cls._discover_operational_equivalences_from_gpt_objects_symbolic(gpt_objects, atol=atol)

        objects_raw, _outcome_counts, valid_mask = normalize_grouped_vectors_single_setting_many_outcomes(
            gpt_objects,
            name="gpt_objects",
        )
        raw = np.asarray(objects_raw, dtype=complex)
        if raw.size and np.max(np.abs(np.imag(raw))) > float(atol):
            raise ValueError("gpt_objects contains significant imaginary components.")
        objects = np.real(raw).astype(float, copy=False)

        num_s, num_o, vec_dim = objects.shape
        matrix_full = objects.reshape(num_s * num_o, vec_dim)
        valid_indices = flatten_valid_indices(valid_mask)
        matrix = matrix_full[valid_indices, :]
        basis_reduced = null_space_basis(matrix.T, atol=atol)
        basis_full = embed_basis_rows_to_padded(
            basis_reduced,
            valid_flat_indices=valid_indices,
            total_size=num_s * num_o,
        )
        basis = basis_full.reshape(-1, num_s, num_o)

        invalid = ~valid_mask[np.newaxis, :, :]
        if invalid.any():
            basis = np.where(invalid, 0.0, basis)
        return basis

    @classmethod
    def infer_measurements_from_gpt_effect_set(
        cls,
        gpt_effect_set: np.ndarray,
        unit_effect: np.ndarray | None = None,
        atol: float = 1e-9,
        outcomes_per_measurement: int | None = None,
    ) -> tuple[np.ndarray, list[tuple[int, ...]]]:
        """Infer measurements as subsets of effects that sum to the unit effect."""
        effects = np.asarray(gpt_effect_set, dtype=complex)
        if effects.ndim != 2:
            raise ValueError("gpt_effect_set must have shape (N_effects, K).")

        n_effects, k = effects.shape
        if unit_effect is None:
            d_float = np.sqrt(k)
            d = int(d_float)
            if d * d != k:
                raise ValueError("Cannot infer unit effect from K. Provide unit_effect explicitly.")
            unit = cls.unit_effect_vector(d).astype(complex)
        else:
            unit = np.asarray(unit_effect, dtype=complex)
            if unit.shape != (k,):
                raise ValueError(f"unit_effect must have shape ({k},).")

        if outcomes_per_measurement is None:
            candidate_sizes = range(1, n_effects + 1)
        else:
            if outcomes_per_measurement <= 0 or outcomes_per_measurement > n_effects:
                raise ValueError("Invalid outcomes_per_measurement.")
            candidate_sizes = [outcomes_per_measurement]

        measurement_indices: list[tuple[int, ...]] = []
        for size in candidate_sizes:
            for combo in combinations(range(n_effects), size):
                if np.allclose(effects[list(combo)].sum(axis=0), unit, atol=atol):
                    measurement_indices.append(combo)

        if not measurement_indices:
            raise ValueError("No measurement subsets summing to unit effect were found.")

        max_outcomes = max(len(combo) for combo in measurement_indices)
        measurement_effects = np.zeros((len(measurement_indices), max_outcomes, k), dtype=complex)
        for y, combo in enumerate(measurement_indices):
            for b, effect_idx in enumerate(combo):
                measurement_effects[y, b, :] = effects[effect_idx]
        return measurement_effects, measurement_indices

    @classmethod
    def data_table_from_gpt_states_and_effect_set(
        cls,
        gpt_states: np.ndarray,
        gpt_effect_set: np.ndarray,
        source_outcome_distribution: np.ndarray | None = None,
        unit_effect: np.ndarray | None = None,
        atol: float = 1e-9,
        outcomes_per_measurement: int | None = None,
        drop_tiny_imag: bool = True,
    ) -> tuple[np.ndarray, list[tuple[int, ...]]]:
        """Build joint ``P(AB|XY)`` from GPT states and a flat GPT effect set."""
        measurement_effects, measurement_indices = cls.infer_measurements_from_gpt_effect_set(
            gpt_effect_set=gpt_effect_set,
            unit_effect=unit_effect,
            atol=atol,
            outcomes_per_measurement=outcomes_per_measurement,
        )
        joint = cls.probability_table_from_gpt_vectors(
            gpt_states,
            measurement_effects,
            source_outcome_distribution=source_outcome_distribution,
            normalize_source_outcomes=True,
            atol=atol,
            drop_tiny_imag=drop_tiny_imag,
        )
        return joint, measurement_indices

    @staticmethod
    def _freeze_object_array(values: object) -> np.ndarray:
        arr = np.asarray(values, dtype=object).copy()
        arr.setflags(write=False)
        return arr

    @property
    def gpt_states_grouped(self) -> np.ndarray:
        return np.asarray(self._gpt_states_grouped_data, dtype=object)

    @property
    def gpt_effects_grouped(self) -> np.ndarray:
        return np.asarray(self._gpt_effects_grouped_data, dtype=object)

    @property
    def measurement_indices(self) -> tuple[tuple[int, ...], ...]:
        if self._measurement_indices_override is not None:
            return tuple(tuple(int(entry) for entry in row) for row in self._measurement_indices_override)
        return tuple(tuple(int(entry) for entry in row) for row in self._measurement_indices_data)

    def _resolve_gpt_inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[int, ...], ...]]:
        if self._input_gpt_states is None or self._input_gpt_effects is None:
            raise ValueError("GPT inputs are unavailable for this scenario.")

        states_flat = self._coerce_flat_vector_set(self._input_gpt_states, name="gpt_states")
        states_grouped = states_flat[:, np.newaxis, :]
        self._assert_machine_numeric_real(states_grouped, name="gpt_states", atol=self.atol)

        if self._input_measurement_indices is not None:
            effects_grouped, measurement_indices_resolved = self._resolve_grouped_gpt_vectors(
                gpt_values=self._input_gpt_effects,
                grouped_indices=self._input_measurement_indices,
                name="gpt_effects",
            )
        elif self._input_infer_measurement_indices:
            effects_flat = self._coerce_flat_vector_set(self._input_gpt_effects, name="gpt_effects")
            self._assert_machine_numeric_real(effects_flat, name="gpt_effects", atol=self.atol)
            effects_grouped, inferred_idx = self.infer_measurements_from_gpt_effect_set(
                gpt_effect_set=effects_flat,
                unit_effect=self._input_unit_effect,
                atol=self.atol,
                outcomes_per_measurement=self._input_outcomes_per_measurement,
            )
            measurement_indices_resolved = tuple(tuple(int(entry) for entry in combo) for combo in inferred_idx)
        else:
            effects_grouped, measurement_indices_resolved = self._resolve_grouped_gpt_vectors(
                gpt_values=self._input_gpt_effects,
                grouped_indices=None,
                name="gpt_effects",
            )
        self._assert_machine_numeric_real(effects_grouped, name="gpt_effects", atol=self.atol)
        return states_flat, states_grouped, effects_grouped, measurement_indices_resolved

    def print_measurement_index_sets(
        self,
        measurement_indices: Sequence[Sequence[int]] | None = None,
    ) -> None:
        """Print measurement index sets for GPT/quantum grouped demos."""
        indices = self.measurement_indices if measurement_indices is None else measurement_indices
        print("\nProvided measurement index sets:")
        for y, idx in enumerate(indices):
            print(f"y={y}: effects {tuple(idx)}")


class QuantumContextualityScenario(GPTContextualityScenario):
    """Contextuality scenario built from quantum states/effects with one state per x."""

    quantum_states_grouped: tuple[tuple[np.ndarray, ...], ...]
    quantum_effects_grouped: tuple[tuple[np.ndarray, ...], ...]
    basis_used: np.ndarray | None
    unit_effect_used: np.ndarray | None
    used_projector_fast_path: bool

    def __init__(
        self,
        quantum_states: object,
        quantum_effects: object,
        *,
        measurement_indices: Sequence[Sequence[int]] | None = None,
        infer_measurement_indices: bool = False,
        basis: np.ndarray | None = None,
        unit_effect: np.ndarray | None = None,
        atol: float = 1e-9,
        outcomes_per_measurement: int | None = None,
        drop_tiny_imag: bool = True,
        verbose: bool = False,
    ) -> None:
        if measurement_indices is not None and infer_measurement_indices:
            raise ValueError(
                "Provide measurement_indices or set infer_measurement_indices=True, not both."
            )

        states_flat_quantum = self._coerce_flat_matrix_set(
            quantum_states,
            name="quantum_states",
        )

        if measurement_indices is not None:
            effects_grouped_quantum, measurement_indices_resolved = self._resolve_grouped_quantum_matrices(
                quantum_values=quantum_effects,
                grouped_indices=measurement_indices,
                name="quantum_effects",
            )
            effects_flat_for_checks = self._flatten_grouped_quantum_matrices(
                effects_grouped_quantum,
                name="quantum_effects",
            )
            quantum_effects_flat = None
        elif infer_measurement_indices:
            quantum_effects_flat = self._coerce_flat_matrix_set(
                quantum_effects,
                name="quantum_effects",
            )
            effects_flat_for_checks = quantum_effects_flat
            effects_grouped_quantum = tuple()
            measurement_indices_resolved = tuple()
        else:
            effects_grouped_quantum, measurement_indices_resolved = self._resolve_grouped_quantum_matrices(
                quantum_values=quantum_effects,
                grouped_indices=None,
                name="quantum_effects",
            )
            effects_flat_for_checks = self._flatten_grouped_quantum_matrices(
                effects_grouped_quantum,
                name="quantum_effects",
            )
            quantum_effects_flat = None

        if states_flat_quantum.shape[-1] != effects_flat_for_checks.shape[-1]:
            raise ValueError("quantum_states and quantum_effects matrix dimensions must match.")
        d = states_flat_quantum.shape[-1]
        use_projector_fast_path = (
            basis is None
            and unit_effect is None
            and np.max(np.abs(np.imag(states_flat_quantum))) <= float(atol)
            and np.max(np.abs(np.imag(effects_flat_for_checks))) <= float(atol)
            and self._all_projectors(states_flat_quantum, atol=atol)
            and self._all_projectors(effects_flat_for_checks, atol=atol)
        )

        gpt_states_flat = self._convert_flat_quantum_matrix_set_to_gpt(
            flat_quantum=states_flat_quantum,
            basis=basis,
            drop_tiny_imag=drop_tiny_imag,
            use_projector_fast_path=use_projector_fast_path,
        )
        self._assert_machine_numeric_real(gpt_states_flat, name="gpt_states", atol=atol)

        unit_effect_for_inference = unit_effect
        if use_projector_fast_path and unit_effect_for_inference is None:
            unit_effect_for_inference = self._matrix_to_hs_vector(
                np.eye(d, dtype=complex),
                drop_tiny_imag=drop_tiny_imag,
            )

        if infer_measurement_indices:
            assert quantum_effects_flat is not None
            gpt_effects_flat = self._convert_flat_quantum_matrix_set_to_gpt(
                flat_quantum=quantum_effects_flat,
                basis=basis,
                drop_tiny_imag=drop_tiny_imag,
                use_projector_fast_path=use_projector_fast_path,
            )
            self._assert_machine_numeric_real(gpt_effects_flat, name="gpt_effects", atol=atol)
            gpt_effects_grouped, inferred_idx = self.infer_measurements_from_gpt_effect_set(
                gpt_effect_set=gpt_effects_flat,
                unit_effect=unit_effect_for_inference,
                atol=atol,
                outcomes_per_measurement=outcomes_per_measurement,
            )
            measurement_indices_resolved = tuple(tuple(int(entry) for entry in combo) for combo in inferred_idx)
            effects_grouped_quantum = self._group_flat_matrices_by_indices(
                flat_matrices=quantum_effects_flat,
                grouped_indices=measurement_indices_resolved,
            )
        else:
            gpt_effects_grouped = self._convert_grouped_quantum_matrices_to_gpt(
                grouped_quantum=effects_grouped_quantum,
                basis=basis,
                drop_tiny_imag=drop_tiny_imag,
                use_projector_fast_path=use_projector_fast_path,
            )
            self._assert_machine_numeric_real(gpt_effects_grouped, name="gpt_effects", atol=atol)

        super().__init__(
            gpt_states=gpt_states_flat,
            gpt_effects=gpt_effects_grouped,
            measurement_indices=None,
            infer_measurement_indices=False,
            unit_effect=unit_effect_for_inference,
            atol=atol,
            outcomes_per_measurement=outcomes_per_measurement,
            drop_tiny_imag=drop_tiny_imag,
            verbose=verbose,
        )

        self._measurement_indices_override = measurement_indices_resolved
        self.quantum_states_grouped = tuple(
            (self._freeze_numeric_array(states_flat_quantum[x], dtype=complex),)
            for x in range(states_flat_quantum.shape[0])
        )
        self.quantum_effects_grouped = tuple(
            tuple(self._freeze_numeric_array(matrix, dtype=complex) for matrix in setting)
            for setting in effects_grouped_quantum
        )
        self.basis_used = None if basis is None else self._freeze_numeric_array(basis, dtype=complex)
        self.unit_effect_used = (
            None
            if unit_effect_for_inference is None
            else self._freeze_numeric_array(unit_effect_for_inference, dtype=complex)
        )
        self.used_projector_fast_path = bool(use_projector_fast_path)

        if verbose and use_projector_fast_path:
            print("Using projector Hilbert-Schmidt vectorization fast path.")
        if verbose and infer_measurement_indices:
            print("\nInferred measurement index sets:")
            for y, idx in enumerate(self.measurement_indices):
                print(f"y={y}: effects {idx}")

    @classmethod
    def from_quantum_states_effects(
        cls,
        *,
        quantum_states: object,
        quantum_effects: object,
        measurement_indices: Sequence[Sequence[int]] | None = None,
        infer_measurement_indices: bool = False,
        basis: np.ndarray | None = None,
        unit_effect: np.ndarray | None = None,
        atol: float = 1e-9,
        outcomes_per_measurement: int | None = None,
        drop_tiny_imag: bool = True,
        verbose: bool = False,
    ) -> "QuantumContextualityScenario":
        """Factory constructor for quantum-state/effect scenarios."""
        return cls(
            quantum_states=quantum_states,
            quantum_effects=quantum_effects,
            measurement_indices=measurement_indices,
            infer_measurement_indices=infer_measurement_indices,
            basis=basis,
            unit_effect=unit_effect,
            atol=atol,
            outcomes_per_measurement=outcomes_per_measurement,
            drop_tiny_imag=drop_tiny_imag,
            verbose=verbose,
        )

    @classmethod
    def projector(cls, ket: object, drop_tiny_imag: bool = True) -> np.ndarray:
        """Return rank-1 projector ``|psi><psi|`` for a state vector ``ket``."""
        if cls._does_not_contain_floats(ket):
            return cls._projector_from_ket_sympy(ket, drop_tiny_imag=drop_tiny_imag)
        return cls._projector_from_ket_numpy(ket, drop_tiny_imag=drop_tiny_imag)

    @staticmethod
    def _hs_normalize(matrix: np.ndarray) -> np.ndarray:
        norm = np.sqrt(np.trace(matrix @ matrix.conj().T).real)
        if norm <= 0:
            raise ValueError("Cannot normalize zero matrix.")
        return matrix / norm

    @classmethod
    def gell_mann_matrices(cls, d: int) -> np.ndarray:
        """Return normalized Gell-Mann basis with shape ``(d^2, d, d)``."""
        if d <= 0:
            raise ValueError("Dimension d must be positive.")

        basis: list[np.ndarray] = []
        basis.append(cls._hs_normalize(np.eye(d, dtype=complex)))

        for k in range(1, d):
            for j in range(k):
                mat = np.zeros((d, d), dtype=complex)
                mat[j, k] = 1.0
                mat[k, j] = 1.0
                basis.append(cls._hs_normalize(mat))

        for k in range(1, d):
            for j in range(k):
                mat = np.zeros((d, d), dtype=complex)
                mat[j, k] = -1.0j
                mat[k, j] = 1.0j
                basis.append(cls._hs_normalize(mat))

        for ell in range(1, d):
            mat = np.zeros((d, d), dtype=complex)
            mat[np.arange(ell), np.arange(ell)] = 1.0
            mat[ell, ell] = -float(ell)
            basis.append(cls._hs_normalize(mat))

        return np.stack(basis, axis=0)

    @classmethod
    def _validate_or_build_basis(cls, basis: np.ndarray | None, d: int) -> np.ndarray:
        if basis is None:
            return cls.gell_mann_matrices(d)
        gm = np.asarray(basis, dtype=complex)
        if gm.ndim != 3 or gm.shape != (d * d, d, d):
            raise ValueError(f"basis must have shape ({d*d}, {d}, {d}).")
        return gm

    @classmethod
    def convert_matrix_list_to_vector_list(
        cls,
        list_of_matrices: np.ndarray,
        basis: np.ndarray | None = None,
        drop_tiny_imag: bool = True,
    ) -> np.ndarray:
        """Convert matrix objects to GPT vectors via ``v_i = Tr(M G_i)``."""
        mats = np.asarray(list_of_matrices, dtype=complex)
        if mats.ndim < 2 or mats.shape[-2] != mats.shape[-1]:
            raise ValueError("list_of_matrices must have shape (..., d, d).")

        d = mats.shape[-1]
        gm = cls._validate_or_build_basis(basis, d)
        vectors = np.einsum("...ij,kji->...k", mats, gm)
        return np.real_if_close(vectors) if drop_tiny_imag else vectors

    @classmethod
    def convert_matrix_to_vector(
        cls,
        mat: np.ndarray,
        basis: np.ndarray | None = None,
        drop_tiny_imag: bool = True,
    ) -> np.ndarray:
        """Convert one matrix to one GPT vector."""
        return cls.convert_matrix_list_to_vector_list(
            np.asarray(mat, dtype=complex)[np.newaxis, ...],
            basis=basis,
            drop_tiny_imag=drop_tiny_imag,
        )[0]

    @classmethod
    def direct_probability_table_from_quantum(
        cls,
        quantum_states: np.ndarray,
        quantum_effects: np.ndarray,
        drop_tiny_imag: bool = True,
    ) -> np.ndarray:
        """Compute ``P(a,b|x,y)`` directly from matrices via ``Tr(rho E)``."""
        states = np.asarray(quantum_states, dtype=complex)
        effects = np.asarray(quantum_effects, dtype=complex)
        if states.ndim != 4:
            raise ValueError("quantum_states must have shape (X, A, d, d).")
        if effects.ndim != 4:
            raise ValueError("quantum_effects must have shape (Y, B, d, d).")
        if states.shape[-2] != states.shape[-1]:
            raise ValueError("quantum_states matrices must be square.")
        if effects.shape[-2] != effects.shape[-1]:
            raise ValueError("quantum_effects matrices must be square.")
        if states.shape[-1] != effects.shape[-1]:
            raise ValueError("State/effect matrix dimensions do not match.")

        probs = np.einsum("xaij,ybji->xyab", states, effects)
        return np.real_if_close(probs) if drop_tiny_imag else probs

    @classmethod
    def probability_table_from_quantum_via_gpt(
        cls,
        quantum_states: np.ndarray,
        quantum_effects: np.ndarray,
        basis: np.ndarray | None = None,
        drop_tiny_imag: bool = True,
    ) -> np.ndarray:
        """Compute ``P(a,b|x,y)`` by converting matrices to GPT vectors first."""
        states = np.asarray(quantum_states, dtype=complex)
        effects = np.asarray(quantum_effects, dtype=complex)
        if states.ndim != 4 or effects.ndim != 4:
            raise ValueError("Expected states (X,A,d,d) and effects (Y,B,d,d).")

        gpt_states = cls.convert_matrix_list_to_vector_list(states, basis=basis, drop_tiny_imag=drop_tiny_imag)
        gpt_effects = cls.convert_matrix_list_to_vector_list(effects, basis=basis, drop_tiny_imag=drop_tiny_imag)
        return cls.probability_table_from_gpt_vectors(
            gpt_states=gpt_states,
            gpt_effects=gpt_effects,
            drop_tiny_imag=drop_tiny_imag,
        )

    @classmethod
    def discover_operational_equivalences_from_quantum_states(
        cls,
        quantum_states: np.ndarray,
        basis: np.ndarray | None = None,
        atol: float = 1e-9,
        drop_tiny_imag: bool = True,
    ) -> np.ndarray:
        """Discover OPEQs from a generic list of quantum states."""
        mats = np.asarray(quantum_states, dtype=complex)
        if mats.ndim != 3 or mats.shape[-2] != mats.shape[-1]:
            raise ValueError("quantum_states must have shape (X, d, d).")

        gpt_states = cls.convert_matrix_list_to_vector_list(mats, basis=basis, drop_tiny_imag=drop_tiny_imag)
        return cls.discover_operational_equivalences_from_gpt_objects(gpt_states[:, np.newaxis, :], atol=atol)

    @staticmethod
    def _matrix_list_to_hs_vectors(mats: np.ndarray, drop_tiny_imag: bool = True) -> np.ndarray:
        """Vectorize matrix arrays ``(..., d, d)`` to ``(..., d*d)`` via ``vec(M^T)``."""
        arr = np.asarray(mats, dtype=complex)
        if arr.ndim < 2 or arr.shape[-2] != arr.shape[-1]:
            raise ValueError("mats must have shape (..., d, d).")
        vecs = np.swapaxes(arr, -2, -1).reshape(arr.shape[:-2] + (arr.shape[-1] * arr.shape[-1],))
        return np.real_if_close(vecs) if drop_tiny_imag else vecs

    @staticmethod
    def _all_projectors(mats: np.ndarray, atol: float) -> bool:
        """Check projector conditions ``P^2=P`` and Hermiticity for all matrices."""
        arr = np.asarray(mats, dtype=complex)
        if arr.ndim < 2 or arr.shape[-2] != arr.shape[-1]:
            return False
        idempotent = np.allclose(arr @ arr, arr, atol=atol, rtol=0.0)
        hermitian = np.allclose(arr, np.swapaxes(arr.conj(), -2, -1), atol=atol, rtol=0.0)
        return bool(idempotent and hermitian)

    @staticmethod
    def _coerce_flat_matrix_set(values: object, *, name: str) -> np.ndarray:
        """Return flat quantum matrices with shape ``(N, d, d)``."""
        mats = np.asarray(values, dtype=complex)
        if mats.ndim != 3 or mats.shape[-2] != mats.shape[-1]:
            raise ValueError(f"{name} must have flat shape (N, d, d).")
        if mats.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one matrix.")
        return mats

    @staticmethod
    def _is_square_matrix_like(value: object) -> bool:
        arr = np.asarray(value, dtype=object)
        return arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] > 0

    @classmethod
    def _coerce_grouped_matrix_settings(cls, values: object, *, name: str) -> tuple[tuple[np.ndarray, ...], ...]:
        """Normalize grouped quantum matrix input to nested tuples."""
        dense: np.ndarray | None = None
        try:
            dense = np.asarray(values, dtype=complex)
        except (TypeError, ValueError):
            dense = None
        if dense is not None:
            if dense.ndim == 4 and dense.shape[-2] == dense.shape[-1]:
                return tuple(
                    tuple(dense[s, o, :, :] for o in range(dense.shape[1]))
                    for s in range(dense.shape[0])
                )
            if dense.ndim == 3 and dense.shape[-2] == dense.shape[-1]:
                return tuple((dense[s, :, :],) for s in range(dense.shape[0]))

        if not isinstance(values, (list, tuple, np.ndarray)):
            raise ValueError(f"{name} must be array-like.")
        settings = list(values)
        if not settings:
            raise ValueError(f"{name} must contain at least one setting.")

        grouped: list[tuple[np.ndarray, ...]] = []
        for s_idx, setting in enumerate(settings):
            if cls._is_square_matrix_like(setting):
                outcomes = [setting]
            else:
                if not isinstance(setting, (list, tuple, np.ndarray)):
                    raise ValueError(f"{name}[{s_idx}] is not list-like.")
                outcomes = list(setting)
                if not outcomes:
                    raise ValueError(f"{name}[{s_idx}] has zero outcomes.")

            row: list[np.ndarray] = []
            for o_idx, matrix in enumerate(outcomes):
                mat = np.asarray(matrix, dtype=complex)
                if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                    raise ValueError(f"{name}[{s_idx}][{o_idx}] is not a square matrix.")
                row.append(mat)
            grouped.append(tuple(row))

        d = grouped[0][0].shape[0]
        for s_idx, row in enumerate(grouped):
            for o_idx, mat in enumerate(row):
                if mat.shape != (d, d):
                    raise ValueError(
                        f"{name}[{s_idx}][{o_idx}] has shape {mat.shape}, expected {(d, d)}."
                    )
        return tuple(grouped)

    @classmethod
    def _group_flat_matrices_by_indices(
        cls,
        flat_matrices: np.ndarray,
        grouped_indices: Sequence[Sequence[int]],
    ) -> tuple[tuple[np.ndarray, ...], ...]:
        """Group a flat ``(N,d,d)`` matrix set by index tuples."""
        grouped: list[tuple[np.ndarray, ...]] = []
        for group in grouped_indices:
            grouped.append(tuple(flat_matrices[int(idx), :, :] for idx in group))
        return tuple(grouped)

    @classmethod
    def _resolve_grouped_quantum_matrices(
        cls,
        quantum_values: object,
        *,
        grouped_indices: Sequence[Sequence[int]] | None,
        name: str,
    ) -> tuple[tuple[tuple[np.ndarray, ...], ...], tuple[tuple[int, ...], ...]]:
        """Resolve grouped quantum matrices and associated index metadata."""
        if grouped_indices is not None:
            flat = cls._coerce_flat_matrix_set(quantum_values, name=name)
            normalized_groups = cls._normalize_index_groups(
                grouped_indices,
                total_items=flat.shape[0],
                name=f"{name}_indices",
            )
            grouped = cls._group_flat_matrices_by_indices(flat_matrices=flat, grouped_indices=normalized_groups)
            return grouped, normalized_groups

        grouped = cls._coerce_grouped_matrix_settings(quantum_values, name=name)
        local_groups = tuple(tuple(range(len(setting))) for setting in grouped)
        return grouped, local_groups

    @staticmethod
    def _flatten_grouped_quantum_matrices(
        grouped_quantum: tuple[tuple[np.ndarray, ...], ...],
        *,
        name: str,
    ) -> np.ndarray:
        """Flatten grouped quantum matrices to ``(N,d,d)`` for checks."""
        flat = [matrix for setting in grouped_quantum for matrix in setting]
        if not flat:
            raise ValueError(f"{name} must contain at least one matrix.")
        return np.stack(flat, axis=0)

    @classmethod
    def _convert_flat_quantum_matrix_set_to_gpt(
        cls,
        flat_quantum: np.ndarray,
        *,
        basis: np.ndarray | None,
        drop_tiny_imag: bool,
        use_projector_fast_path: bool,
    ) -> np.ndarray:
        """Convert a flat quantum matrix set ``(N,d,d)`` to GPT vectors ``(N,K)``."""
        if use_projector_fast_path:
            return cls._matrix_list_to_hs_vectors(flat_quantum, drop_tiny_imag=drop_tiny_imag)
        return cls.convert_matrix_list_to_vector_list(
            flat_quantum,
            basis=basis,
            drop_tiny_imag=drop_tiny_imag,
        )

    @classmethod
    def _convert_single_quantum_matrix_to_gpt(
        cls,
        matrix: np.ndarray,
        *,
        basis: np.ndarray | None,
        drop_tiny_imag: bool,
        use_projector_fast_path: bool,
    ) -> np.ndarray:
        """Convert one quantum matrix to one GPT vector."""
        if use_projector_fast_path:
            return cls._matrix_to_hs_vector(matrix, drop_tiny_imag=drop_tiny_imag)
        return cls.convert_matrix_to_vector(
            matrix,
            basis=basis,
            drop_tiny_imag=drop_tiny_imag,
        )

    @classmethod
    def _convert_grouped_quantum_matrices_to_gpt(
        cls,
        grouped_quantum: tuple[tuple[np.ndarray, ...], ...],
        *,
        basis: np.ndarray | None,
        drop_tiny_imag: bool,
        use_projector_fast_path: bool,
    ) -> np.ndarray:
        """Convert grouped quantum matrices to grouped GPT vectors."""
        return np.array(
            [
                [
                    cls._convert_single_quantum_matrix_to_gpt(
                        matrix,
                        basis=basis,
                        drop_tiny_imag=drop_tiny_imag,
                        use_projector_fast_path=use_projector_fast_path,
                    )
                    for matrix in setting
                ]
                for setting in grouped_quantum
            ],
            dtype=object,
        )

    @staticmethod
    def _freeze_numeric_array(values: object, *, dtype: object = complex) -> np.ndarray:
        arr = np.asarray(values, dtype=dtype).copy()
        arr.setflags(write=False)
        return arr

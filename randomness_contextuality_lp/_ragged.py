"""Helpers for setting-dependent outcome cardinalities with zero padding."""

from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp


def build_valid_outcome_masks(
    a_cardinality_per_x: np.ndarray,
    b_cardinality_per_y: np.ndarray,
    *,
    a_max: int | None = None,
    b_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``valid_a_mask``, ``valid_b_mask``, and ``valid_ab_mask``."""
    a_counts = np.asarray(a_cardinality_per_x, dtype=int).reshape(-1)
    b_counts = np.asarray(b_cardinality_per_y, dtype=int).reshape(-1)
    if np.any(a_counts <= 0):
        raise ValueError("a_cardinality_per_x must contain strictly positive integers.")
    if np.any(b_counts <= 0):
        raise ValueError("b_cardinality_per_y must contain strictly positive integers.")

    max_a = int(np.max(a_counts)) if a_max is None else int(a_max)
    max_b = int(np.max(b_counts)) if b_max is None else int(b_max)
    if max_a <= 0 or max_b <= 0:
        raise ValueError("a_max and b_max must be strictly positive.")
    if np.any(a_counts > max_a):
        raise ValueError("a_cardinality_per_x exceeds a_max.")
    if np.any(b_counts > max_b):
        raise ValueError("b_cardinality_per_y exceeds b_max.")

    valid_a_mask = np.zeros((a_counts.size, max_a), dtype=bool)
    for x, count in enumerate(a_counts):
        valid_a_mask[x, :count] = True

    valid_b_mask = np.zeros((b_counts.size, max_b), dtype=bool)
    for y, count in enumerate(b_counts):
        valid_b_mask[y, :count] = True

    valid_ab_mask = valid_a_mask[:, np.newaxis, :, np.newaxis] & valid_b_mask[np.newaxis, :, np.newaxis, :]
    return valid_a_mask, valid_b_mask, valid_ab_mask


def normalize_behavior_table(
    data: Any,
    *,
    atol: float = 1e-9,
    pad_value: Any = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize dense/ragged/masked ``data[x][y][a][b]`` into one padded 4D array."""
    if np.ma.isMaskedArray(data):
        dense, inferred_a, inferred_b = _normalize_behavior_table_dense_masked(
            data,
            atol=atol,
            pad_value=pad_value,
        )
        num_x, num_y, max_a, max_b = dense.shape
    else:
        raw = np.asarray(data, dtype=object)
        if raw.ndim == 4:
            num_x, num_y, max_a, max_b = raw.shape
            if num_x == 0 or num_y == 0 or max_a == 0 or max_b == 0:
                raise ValueError("data must have nonzero shape in all axes.")
            dense = raw.copy()
            inferred_a, inferred_b = _infer_cardinalities_from_dense_behavior(dense, atol=atol)
        else:
            dense, inferred_a, inferred_b = _normalize_behavior_table_ragged(data, pad_value=pad_value)
            num_x, num_y, max_a, max_b = dense.shape

    if num_x == 0 or num_y == 0 or max_a == 0 or max_b == 0:
        raise ValueError("data must have nonzero shape in all axes.")

    a_counts = inferred_a
    b_counts = inferred_b

    valid_a_mask, valid_b_mask, valid_ab_mask = build_valid_outcome_masks(
        a_counts,
        b_counts,
        a_max=max_a,
        b_max=max_b,
    )
    return dense, a_counts, b_counts, valid_a_mask, valid_b_mask, valid_ab_mask


def _normalize_behavior_table_dense_masked(
    data: Any,
    *,
    atol: float,
    pad_value: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize dense masked-array data and infer cardinalities from the mask."""
    masked = np.ma.asarray(data, dtype=object)
    if masked.ndim != 4:
        raise ValueError("Masked data must have shape (X, Y, A, B).")
    num_x, num_y, max_a, max_b = masked.shape
    if num_x == 0 or num_y == 0 or max_a == 0 or max_b == 0:
        raise ValueError("data must have nonzero shape in all axes.")

    dense = np.asarray(np.ma.getdata(masked), dtype=object).copy()
    mask = np.asarray(np.ma.getmaskarray(masked), dtype=bool)
    dense[mask] = pad_value

    a_counts = np.zeros(num_x, dtype=int)
    for x in range(num_x):
        active = np.array(
            [
                any(not bool(mask[x, y, a, b]) for y in range(num_y) for b in range(max_b))
                for a in range(max_a)
            ],
            dtype=bool,
        )
        if not np.any(active):
            raise ValueError(f"Could not infer a positive A cardinality for x={x} from mask.")
        last = int(np.flatnonzero(active).max())
        if np.any(~active[: last + 1]):
            raise ValueError(
                f"Masked dense data for x={x} has non-suffix padding along A "
                "(interior masked hole before a later active outcome)."
            )
        a_counts[x] = last + 1

    b_counts = np.zeros(num_y, dtype=int)
    for y in range(num_y):
        active = np.array(
            [
                any(not bool(mask[x, y, a, b]) for x in range(num_x) for a in range(int(a_counts[x])))
                for b in range(max_b)
            ],
            dtype=bool,
        )
        if not np.any(active):
            raise ValueError(f"Could not infer a positive B cardinality for y={y} from mask.")
        last = int(np.flatnonzero(active).max())
        if np.any(~active[: last + 1]):
            raise ValueError(
                f"Masked dense data for y={y} has non-suffix padding along B "
                "(interior masked hole before a later active outcome)."
            )
        b_counts[y] = last + 1

    _, _, valid_ab_mask = build_valid_outcome_masks(
        a_counts,
        b_counts,
        a_max=max_a,
        b_max=max_b,
    )
    invalid = ~valid_ab_mask

    if np.any(mask & valid_ab_mask):
        raise ValueError(
            "Masked data has masked entries inside inferred valid support. "
            "Masks are interpreted as structural padding only."
        )

    for idx in np.argwhere(invalid & ~mask):
        i = tuple(idx.tolist())
        if not _is_zero_entry(dense[i], atol=atol):
            raise ValueError("Masked dense data has nonzero entry outside inferred support.")
        dense[i] = pad_value
    dense[invalid] = pad_value
    return dense, a_counts, b_counts


def _infer_cardinalities_from_dense_behavior(
    dense: np.ndarray,
    *,
    atol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer ``A=A(x)`` and ``B=B(y)`` by trailing-zero support in dense data."""
    arr = np.asarray(dense, dtype=object)
    if arr.ndim != 4:
        raise ValueError("dense must have shape (X, Y, A, B).")
    num_x, num_y, max_a, max_b = arr.shape

    a_counts = np.zeros(num_x, dtype=int)
    for x in range(num_x):
        active = np.array(
            [
                any(
                    not _is_zero_entry(arr[x, y, a, b], atol=atol)
                    for y in range(num_y)
                    for b in range(max_b)
                )
                for a in range(max_a)
            ],
            dtype=bool,
        )
        if not np.any(active):
            raise ValueError(f"Could not infer a positive A cardinality for x={x}.")
        last = int(np.flatnonzero(active).max())
        if np.any(~active[: last + 1]):
            raise ValueError(
                f"Dense data for x={x} has non-suffix padding along A "
                "(interior zero hole before a later active outcome). "
                "Use ragged input or a masked array to avoid ambiguity."
            )
        count_for_x = last + 1
        if count_for_x is None or count_for_x <= 0:
            raise ValueError(f"Could not infer a positive A cardinality for x={x}.")
        a_counts[x] = count_for_x

    b_counts = np.zeros(num_y, dtype=int)
    for y in range(num_y):
        active = np.array(
            [
                any(
                    not _is_zero_entry(arr[x, y, a, b], atol=atol)
                    for x in range(num_x)
                    for a in range(int(a_counts[x]))
                )
                for b in range(max_b)
            ],
            dtype=bool,
        )
        if not np.any(active):
            raise ValueError(f"Could not infer a positive B cardinality for y={y}.")
        last = int(np.flatnonzero(active).max())
        if np.any(~active[: last + 1]):
            raise ValueError(
                f"Dense data for y={y} has non-suffix padding along B "
                "(interior zero hole before a later active outcome). "
                "Use ragged input or a masked array to avoid ambiguity."
            )
        count_for_y = last + 1
        if count_for_y is None or count_for_y <= 0:
            raise ValueError(f"Could not infer a positive B cardinality for y={y}.")
        b_counts[y] = count_for_y

    # Enforce structural zeros outside inferred support.
    for x in range(num_x):
        for y in range(num_y):
            for a in range(a_counts[x], max_a):
                for b in range(max_b):
                    if not _is_zero_entry(arr[x, y, a, b], atol=atol):
                        raise ValueError("Dense data has nonzero entry outside inferred A support.")
            for a in range(a_counts[x]):
                for b in range(b_counts[y], max_b):
                    if not _is_zero_entry(arr[x, y, a, b], atol=atol):
                        raise ValueError("Dense data has nonzero entry outside inferred B support.")
    return a_counts, b_counts


def _is_zero_entry(value: object, *, atol: float) -> bool:
    """Return whether one scalar entry is numerically/symbolically zero."""
    expr = sp.sympify(value)
    if expr.is_zero is True:
        return True
    try:
        numeric = complex(sp.N(expr))
    except (TypeError, ValueError):
        return False
    return abs(numeric) <= float(atol)


def normalize_opeq_array(
    opeqs: Any,
    *,
    num_settings: int,
    outcome_cardinality_per_setting: np.ndarray,
    max_outcomes: int,
    name: str,
    pad_value: Any = 0,
) -> np.ndarray:
    """Normalize dense/ragged OPEQ input to padded ``(N, S, O_max)``."""
    counts = np.asarray(outcome_cardinality_per_setting, dtype=int).reshape(-1)
    if counts.shape != (num_settings,):
        raise ValueError(f"{name}: outcome_cardinality_per_setting must have shape ({num_settings},).")
    if np.any(counts <= 0):
        raise ValueError(f"{name}: outcome_cardinality_per_setting must be strictly positive.")
    if int(max_outcomes) <= 0:
        raise ValueError(f"{name}: max_outcomes must be positive.")

    arr = np.asarray(opeqs, dtype=object)
    if arr.ndim == 2:
        if arr.shape[0] != num_settings:
            raise ValueError(f"{name}: single OPEQ must have first axis size {num_settings}.")
        if arr.shape[1] != max_outcomes:
            raise ValueError(
                f"{name}: dense OPEQ input must use padded outcome size {max_outcomes}; "
                "use ragged list input for per-setting cardinalities."
            )
        return arr[np.newaxis, :, :]
    if arr.ndim == 3:
        if arr.shape[1] != num_settings:
            raise ValueError(f"{name}: OPEQ list must have second axis size {num_settings}.")
        if arr.shape[2] != max_outcomes:
            raise ValueError(
                f"{name}: dense OPEQ input must use padded outcome size {max_outcomes}; "
                "use ragged list input for per-setting cardinalities."
            )
        return arr

    if not _is_sequence_like(opeqs):
        raise ValueError(f"{name}: expected dense array or nested list-like input.")
    root = _as_list(opeqs)
    if len(root) == 0:
        return np.zeros((0, num_settings, max_outcomes), dtype=object)

    if _looks_like_single_opeq(root, num_settings=num_settings):
        eq_list = [root]
    else:
        eq_list = []
        for idx, eq in enumerate(root):
            eq_rows = _as_list(eq)
            if not _looks_like_single_opeq(eq_rows, num_settings=num_settings):
                raise ValueError(f"{name}[{idx}] must describe one OPEQ with {num_settings} settings.")
            eq_list.append(eq_rows)

    out = np.empty((len(eq_list), num_settings, max_outcomes), dtype=object)
    out[:, :, :] = pad_value
    for n, eq in enumerate(eq_list):
        for s in range(num_settings):
            row = _as_list(eq[s])
            expected = int(counts[s])
            if len(row) == expected:
                for o in range(expected):
                    out[n, s, o] = row[o]
                continue
            if len(row) == max_outcomes:
                for o in range(max_outcomes):
                    out[n, s, o] = row[o]
                continue
            raise ValueError(
                f"{name}: setting {s} expects {expected} outcomes (or padded length {max_outcomes}), "
                f"got {len(row)}."
            )
    return out


def structural_zero_opeqs(
    *,
    num_settings: int,
    max_outcomes: int,
    outcome_cardinality_per_setting: np.ndarray,
) -> np.ndarray:
    """Return OPEQs enforcing all padded coordinates to be identically zero."""
    counts = np.asarray(outcome_cardinality_per_setting, dtype=int).reshape(-1)
    if counts.shape != (num_settings,):
        raise ValueError("outcome_cardinality_per_setting has incompatible shape.")
    zero_rows: list[np.ndarray] = []
    for setting in range(num_settings):
        count = int(counts[setting])
        for outcome in range(count, max_outcomes):
            eq = np.zeros((num_settings, max_outcomes), dtype=float)
            eq[setting, outcome] = 1.0
            zero_rows.append(eq)
    if not zero_rows:
        return np.zeros((0, num_settings, max_outcomes), dtype=float)
    return np.stack(zero_rows, axis=0)


def flatten_valid_indices(valid_mask: np.ndarray) -> np.ndarray:
    """Flatten a ``(S, O_max)`` validity mask to 1D indices."""
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("valid_mask must be 2D.")
    return np.flatnonzero(mask.reshape(-1))


def embed_basis_rows_to_padded(
    basis_rows: np.ndarray,
    *,
    valid_flat_indices: np.ndarray,
    total_size: int,
) -> np.ndarray:
    """Embed reduced-coordinate basis rows into padded coordinates."""
    basis = np.asarray(basis_rows)
    if basis.ndim == 1:
        basis = basis[np.newaxis, :]
    if basis.ndim != 2:
        raise ValueError("basis_rows must be 1D or 2D.")
    valid_idx = np.asarray(valid_flat_indices, dtype=int).reshape(-1)
    if basis.shape[1] != valid_idx.size:
        raise ValueError("basis_rows second dimension must match number of valid indices.")
    if np.any(valid_idx < 0) or np.any(valid_idx >= int(total_size)):
        raise ValueError("valid_flat_indices are out of bounds.")

    dtype = object if basis.dtype == object else basis.dtype
    out = np.zeros((basis.shape[0], int(total_size)), dtype=dtype)
    out[:, valid_idx] = basis
    return out


def normalize_grouped_vectors_settings_single_outcome(
    values: Any,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize grouped vectors where 2D means ``(S, K)`` (one outcome per setting)."""
    arr = np.asarray(values, dtype=object)
    if arr.ndim == 2:
        promoted = arr[:, np.newaxis, :]
        counts = np.ones(promoted.shape[0], dtype=int)
        valid = np.ones((promoted.shape[0], 1), dtype=bool)
        return promoted, counts, valid
    if arr.ndim == 3:
        counts = np.full(arr.shape[0], arr.shape[1], dtype=int)
        valid = np.ones((arr.shape[0], arr.shape[1]), dtype=bool)
        return arr, counts, valid
    return _normalize_grouped_vectors_ragged(values, name=name)


def normalize_grouped_vectors_single_setting_many_outcomes(
    values: Any,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize grouped vectors where 2D means ``(O, K)`` for one setting."""
    arr = np.asarray(values, dtype=object)
    if arr.ndim == 2:
        promoted = arr[np.newaxis, :, :]
        counts = np.array([promoted.shape[1]], dtype=int)
        valid = np.ones((1, promoted.shape[1]), dtype=bool)
        return promoted, counts, valid
    if arr.ndim == 3:
        counts = np.full(arr.shape[0], arr.shape[1], dtype=int)
        valid = np.ones((arr.shape[0], arr.shape[1]), dtype=bool)
        return arr, counts, valid
    return _normalize_grouped_vectors_ragged(values, name=name)


def _normalize_behavior_table_ragged(
    data: Any,
    *,
    pad_value: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_blocks = _as_list(data)
    if len(x_blocks) == 0:
        raise ValueError("data must have at least one preparation setting x.")
    y0 = _as_list(x_blocks[0])
    if len(y0) == 0:
        raise ValueError("data must have at least one measurement setting y.")
    num_x = len(x_blocks)
    num_y = len(y0)

    for x, block in enumerate(x_blocks):
        y_blocks = _as_list(block)
        if len(y_blocks) != num_y:
            raise ValueError(f"data[{x}] has {len(y_blocks)} y-settings, expected {num_y}.")

    a_counts = np.zeros(num_x, dtype=int)
    for x in range(num_x):
        a_expected: int | None = None
        for y in range(num_y):
            outcomes_a = _as_list(_as_list(x_blocks[x])[y])
            if a_expected is None:
                a_expected = len(outcomes_a)
                if a_expected == 0:
                    raise ValueError(f"data[{x}][{y}] has zero a-outcomes.")
            elif len(outcomes_a) != a_expected:
                raise ValueError(
                    f"Ragged data violates A=A(x): data[{x}][{y}] has {len(outcomes_a)} a-outcomes, "
                    f"expected {a_expected} for x={x}."
                )
        a_counts[x] = int(a_expected)

    b_counts = np.zeros(num_y, dtype=int)
    for y in range(num_y):
        b_expected: int | None = None
        for x in range(num_x):
            outcomes_a = _as_list(_as_list(x_blocks[x])[y])
            if len(outcomes_a) != a_counts[x]:
                raise ValueError(
                    f"data[{x}][{y}] has {len(outcomes_a)} a-outcomes, expected {a_counts[x]} for x={x}."
                )
            for a in range(a_counts[x]):
                outcomes_b = _as_list(outcomes_a[a])
                if b_expected is None:
                    b_expected = len(outcomes_b)
                    if b_expected == 0:
                        raise ValueError(f"data[{x}][{y}][{a}] has zero b-outcomes.")
                elif len(outcomes_b) != b_expected:
                    raise ValueError(
                        f"Ragged data violates B=B(y): data[{x}][{y}][{a}] has {len(outcomes_b)} b-outcomes, "
                        f"expected {b_expected} for y={y}."
                    )
        b_counts[y] = int(b_expected)

    max_a = int(np.max(a_counts))
    max_b = int(np.max(b_counts))
    dense = np.empty((num_x, num_y, max_a, max_b), dtype=object)
    dense[:, :, :, :] = pad_value
    for x in range(num_x):
        for y in range(num_y):
            outcomes_a = _as_list(_as_list(x_blocks[x])[y])
            for a in range(a_counts[x]):
                outcomes_b = _as_list(outcomes_a[a])
                for b in range(b_counts[y]):
                    dense[x, y, a, b] = outcomes_b[b]
    return dense, a_counts, b_counts


def _normalize_grouped_vectors_ragged(
    values: Any,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    settings = _as_list(values)
    if len(settings) == 0:
        raise ValueError(f"{name} must contain at least one setting.")

    num_settings = len(settings)
    counts = np.zeros(num_settings, dtype=int)
    vec_dim: int | None = None
    for setting_idx, setting in enumerate(settings):
        outcomes = _as_list(setting)
        if len(outcomes) == 0:
            raise ValueError(f"{name}[{setting_idx}] has zero outcomes.")
        counts[setting_idx] = len(outcomes)
        for outcome_idx, outcome in enumerate(outcomes):
            vector = _as_list(outcome)
            if len(vector) == 0:
                raise ValueError(f"{name}[{setting_idx}][{outcome_idx}] has zero vector dimension.")
            if vec_dim is None:
                vec_dim = len(vector)
            elif len(vector) != vec_dim:
                raise ValueError(
                    f"{name}[{setting_idx}][{outcome_idx}] has vector dimension {len(vector)}, "
                    f"expected {vec_dim}."
                )

    max_outcomes = int(np.max(counts))
    assert vec_dim is not None
    dense = np.empty((num_settings, max_outcomes, vec_dim), dtype=object)
    dense[:, :, :] = 0
    for setting_idx in range(num_settings):
        outcomes = _as_list(settings[setting_idx])
        for outcome_idx, outcome in enumerate(outcomes):
            vector = _as_list(outcome)
            for k in range(vec_dim):
                dense[setting_idx, outcome_idx, k] = vector[k]

    valid = np.zeros((num_settings, max_outcomes), dtype=bool)
    for setting_idx, count in enumerate(counts):
        valid[setting_idx, :count] = True
    return dense, counts, valid


def _looks_like_single_opeq(value: list[Any], *, num_settings: int) -> bool:
    if len(value) != num_settings:
        return False
    for row in value:
        if not _is_sequence_like(row):
            return False
        row_seq = _as_list(row)
        if any(_is_sequence_like(entry) for entry in row_seq):
            return False
    return True


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        return list(value)
    raise ValueError("Expected a list/tuple/ndarray nested structure.")


def _is_sequence_like(value: Any) -> bool:
    if isinstance(value, (list, tuple, np.ndarray)):
        return True
    if isinstance(value, (str, bytes)):
        return False
    return hasattr(value, "__iter__")

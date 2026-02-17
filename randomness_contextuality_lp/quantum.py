"""Quantum and GPT utilities for contextuality workflows."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .linalg_utils import null_space_basis
from .scenario import ContextualityScenario


# ============================================================================
# Quantum-specific functions
# ============================================================================

def projector(ket: object, drop_tiny_imag: bool = True) -> np.ndarray:
    """Return the rank-1 projector ``|psi><psi|`` for a state vector ``ket``."""
    if _contains_sympy_entries(ket):
        return _projector_from_ket_sympy(ket, drop_tiny_imag=drop_tiny_imag)
    return _projector_from_ket_numpy(ket, drop_tiny_imag=drop_tiny_imag)


def projector_hs_vector(ket: object, drop_tiny_imag: bool = True) -> np.ndarray:
    """Return Hilbert-Schmidt vectorization of ``|psi><psi|``.

    Uses ``vec(P^T)`` so vector inner products match ``Tr(PQ)`` under
    ``np.einsum(...k,...k)`` conventions used in this module.
    """
    if _contains_sympy_entries(ket):
        proj_sym = _projector_from_ket_sympy(ket, drop_tiny_imag=False)
        return _matrix_to_hs_vector_sympy(proj_sym, drop_tiny_imag=drop_tiny_imag)
    proj_num = _projector_from_ket_numpy(ket, drop_tiny_imag=False)
    return _matrix_to_hs_vector(proj_num, drop_tiny_imag=drop_tiny_imag)


def gell_mann_matrices(d: int) -> np.ndarray:
    """Return normalized Gell-Mann basis with shape ``(d^2, d, d)``."""
    if d <= 0:
        raise ValueError("Dimension d must be positive.")

    basis: list[np.ndarray] = []
    basis.append(_hs_normalize(np.eye(d, dtype=complex)))

    for k in range(1, d):
        for j in range(k):
            mat = np.zeros((d, d), dtype=complex)
            mat[j, k] = 1.0
            mat[k, j] = 1.0
            basis.append(_hs_normalize(mat))

    for k in range(1, d):
        for j in range(k):
            mat = np.zeros((d, d), dtype=complex)
            mat[j, k] = -1.0j
            mat[k, j] = 1.0j
            basis.append(_hs_normalize(mat))

    for ell in range(1, d):
        mat = np.zeros((d, d), dtype=complex)
        mat[np.arange(ell), np.arange(ell)] = 1.0
        mat[ell, ell] = -float(ell)
        basis.append(_hs_normalize(mat))

    return np.stack(basis, axis=0)


def convert_matrix_list_to_vector_list(
    list_of_matrices: np.ndarray,
    basis: np.ndarray | None = None,
    drop_tiny_imag: bool = True,
) -> np.ndarray:
    """Convert matrix objects to GPT vectors via ``v_i = Tr(M G_i)``."""
    mats = np.asarray(list_of_matrices, dtype=complex)
    if mats.ndim < 2 or mats.shape[-2] != mats.shape[-1]:
        raise ValueError("list_of_matrices must have shape (..., d, d).")

    d = mats.shape[-1]
    gm = _validate_or_build_basis(basis, d)
    vectors = np.einsum("...ij,kji->...k", mats, gm)
    return np.real_if_close(vectors) if drop_tiny_imag else vectors


def convert_matrix_to_vector(
    mat: np.ndarray,
    basis: np.ndarray | None = None,
    drop_tiny_imag: bool = True,
) -> np.ndarray:
    """Convert one matrix to one GPT vector."""
    return convert_matrix_list_to_vector_list(
        np.asarray(mat, dtype=complex)[np.newaxis, ...],
        basis=basis,
        drop_tiny_imag=drop_tiny_imag,
    )[0]


def direct_probability_table_from_quantum(
    quantum_states: np.ndarray,
    quantum_effects: np.ndarray,
    drop_tiny_imag: bool = True,
) -> np.ndarray:
    """Compute ``P(a,b|x,y)`` directly from matrices via ``Tr(rho E)``.

    Parameters
    ----------
    quantum_states:
        Shape ``(X, A, d, d)``.
    quantum_effects:
        Shape ``(Y, B, d, d)``.
    """
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


def probability_table_from_quantum_via_gpt(
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

    gpt_states = convert_matrix_list_to_vector_list(states, basis=basis, drop_tiny_imag=drop_tiny_imag)
    gpt_effects = convert_matrix_list_to_vector_list(effects, basis=basis, drop_tiny_imag=drop_tiny_imag)
    return probability_table_from_gpt_vectors(gpt_states=gpt_states, gpt_effects=gpt_effects, drop_tiny_imag=drop_tiny_imag)


# ============================================================================
# GPT-specific functions
# ============================================================================

def maximally_mixed_vector(d: int) -> np.ndarray:
    """Return GPT vector for maximally mixed state ``I/d`` in this basis."""
    vec = np.zeros(d * d, dtype=float)
    vec[0] = 1.0 / np.sqrt(d)
    return vec


def unit_effect_vector(d: int) -> np.ndarray:
    """Return GPT vector for unit effect ``I`` in this basis."""
    vec = np.zeros(d * d, dtype=float)
    vec[0] = np.sqrt(d)
    return vec


def probability_table_from_gpt_vectors(
    gpt_states: np.ndarray,
    gpt_effects: np.ndarray,
    source_outcome_distribution: np.ndarray | None = None,
    normalize_source_outcomes: bool = True,
    atol: float = 1e-9,
    drop_tiny_imag: bool = True,
) -> np.ndarray:
    """Compute GPT probability table from grouped state/effect vectors.

    By default this returns a joint table ``P(a,b|x,y)`` by weighting
    ``p(b|x,y,a)`` with ``P(a|x)``. If ``source_outcome_distribution`` is not
    provided, a uniform distribution over ``a`` is used for each ``x``.

    Set ``normalize_source_outcomes=False`` to return the conditional table
    ``p(b|x,y,a)`` directly.
    """
    if _contains_sympy_entries(gpt_states) or _contains_sympy_entries(gpt_effects):
        return _probability_table_from_gpt_vectors_symbolic(
            gpt_states=gpt_states,
            gpt_effects=gpt_effects,
            source_outcome_distribution=source_outcome_distribution,
            normalize_source_outcomes=normalize_source_outcomes,
            atol=atol,
            drop_tiny_imag=drop_tiny_imag,
        )

    states = np.asarray(gpt_states, dtype=complex)
    effects = np.asarray(gpt_effects, dtype=complex)
    if states.ndim != 3:
        raise ValueError("gpt_states must have shape (X, A, K).")
    if effects.ndim != 3:
        raise ValueError("gpt_effects must have shape (Y, B, K).")
    if states.shape[-1] != effects.shape[-1]:
        raise ValueError("State/effect vector dimensions do not match.")

    probs = np.einsum("xak,ybk->xyab", states, effects)
    if normalize_source_outcomes:
        num_x, _, num_a, _ = probs.shape
        if source_outcome_distribution is None:
            p_a_given_x = np.full((num_x, num_a), 1.0 / float(num_a), dtype=float)
        else:
            p_a_given_x = np.asarray(source_outcome_distribution, dtype=float)
            if p_a_given_x.shape != (num_x, num_a):
                raise ValueError(f"source_outcome_distribution must have shape ({num_x}, {num_a}).")
            if np.any(p_a_given_x < -atol):
                raise ValueError("source_outcome_distribution contains negative entries.")
            if not np.allclose(p_a_given_x.sum(axis=1), 1.0, atol=atol):
                raise ValueError("Each source_outcome_distribution[x,:] must sum to 1.")
            p_a_given_x = np.clip(p_a_given_x, 0.0, None)
        probs = probs * p_a_given_x[:, np.newaxis, :, np.newaxis]
    return np.real_if_close(probs) if drop_tiny_imag else probs


def discover_operational_equivalences_from_gpt_objects(
    gpt_objects: np.ndarray,
    atol: float = 1e-9,
) -> np.ndarray:
    """Discover operational equivalences from GPT vectors via nullspace.

    Input conventions:
    - 2D input: shape ``(O, K)`` is auto-promoted to ``(1, O, K)``.
    - 3D input: shape ``(S, O, K)``.

    Output:
    - shape ``(N_opeq, S, O)`` (always 3D).
    """
    if _contains_sympy_entries(gpt_objects):
        return _discover_operational_equivalences_from_gpt_objects_symbolic(gpt_objects, atol=atol)

    raw = np.asarray(gpt_objects, dtype=complex)
    if np.max(np.abs(np.imag(raw))) > atol:
        raise ValueError("gpt_objects contains significant imaginary components.")
    objects = np.real_if_close(raw).astype(float)
    if objects.ndim == 2:
        objects = objects[np.newaxis, ...]
    if objects.ndim != 3:
        raise ValueError("gpt_objects must have shape (O,K) or (S,O,K).")

    num_s, num_o, vec_dim = objects.shape
    matrix = objects.reshape(num_s * num_o, vec_dim)
    basis = null_space_basis(matrix.T, atol=atol)
    return basis.reshape(-1, num_s, num_o)


def discover_operational_equivalences_from_quantum_states(
    quantum_states: np.ndarray,
    basis: np.ndarray | None = None,
    atol: float = 1e-9,
    drop_tiny_imag: bool = True,
) -> np.ndarray:
    """Discover OPEQs from a generic list of quantum states.

    Input shape:
    - ``(X, d, d)`` interpreted as ``X`` settings with one outcome ``A=1``.

    Output shape:
    - ``(N_opeq, X, 1)``.
    """
    mats = np.asarray(quantum_states, dtype=complex)
    if mats.ndim != 3 or mats.shape[-2] != mats.shape[-1]:
        raise ValueError("quantum_states must have shape (X, d, d).")

    gpt_states = convert_matrix_list_to_vector_list(mats, basis=basis, drop_tiny_imag=drop_tiny_imag)  # (X, K)
    return discover_operational_equivalences_from_gpt_objects(gpt_states[:, np.newaxis, :], atol=atol)


def infer_measurements_from_gpt_effect_set(
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
        unit = unit_effect_vector(d).astype(complex)
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

    cardinalities = {len(combo) for combo in measurement_indices}
    if len(cardinalities) != 1:
        raise ValueError(
            "Inferred measurements have varying numbers of outcomes; "
            "set outcomes_per_measurement to choose one cardinality."
        )

    measurement_effects = np.stack([effects[list(combo)] for combo in measurement_indices], axis=0)
    return measurement_effects, measurement_indices


def data_table_from_gpt_states_and_effect_set(
    gpt_states: np.ndarray,
    gpt_effect_set: np.ndarray,
    source_outcome_distribution: np.ndarray | None = None,
    unit_effect: np.ndarray | None = None,
    atol: float = 1e-9,
    outcomes_per_measurement: int | None = None,
    drop_tiny_imag: bool = True,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Build joint ``P(AB|XY)`` from GPT states and a flat GPT effect set.

    State conventions:
    - ``(X, K)`` means X settings with one outcome (A=1).
    - ``(X, A, K)`` is used as-is.
    """
    states = np.asarray(gpt_states, dtype=complex)
    if states.ndim == 2:
        states = states[:, np.newaxis, :]
    if states.ndim != 3:
        raise ValueError("gpt_states must have shape (X, K) or (X, A, K).")

    measurement_effects, measurement_indices = infer_measurements_from_gpt_effect_set(
        gpt_effect_set=gpt_effect_set,
        unit_effect=unit_effect,
        atol=atol,
        outcomes_per_measurement=outcomes_per_measurement,
    )
    joint = probability_table_from_gpt_vectors(
        states,
        measurement_effects,
        source_outcome_distribution=source_outcome_distribution,
        normalize_source_outcomes=True,
        atol=atol,
        drop_tiny_imag=drop_tiny_imag,
    )
    return joint, measurement_indices


def contextuality_scenario_from_gpt(
    gpt_states: np.ndarray,
    gpt_effect_set: np.ndarray,
    source_outcome_distribution: np.ndarray | None = None,
    unit_effect: np.ndarray | None = None,
    atol: float = 1e-9,
    outcomes_per_measurement: int | None = None,
    drop_tiny_imag: bool = True,
    verbose: bool = False,
    return_measurement_indices: bool = False,
) -> ContextualityScenario | tuple[ContextualityScenario, list[tuple[int, ...]]]:
    """Build a full ``ContextualityScenario`` from GPT states and a flat effect set.

    Motivation
    ----------
    Use this when your model is already in GPT-vector form and you want the package
    to infer measurement groupings and operational equivalences automatically before
    running randomness LPs.

    How to use it with other functions
    ----------------------------------
    This is the main bridge from GPT object descriptions to optimization. The
    returned ``ContextualityScenario`` is intended to be passed directly into
    ``eve_optimal_guessing_probability`` or
    ``eve_optimal_average_guessing_probability``.

    Input/output structure
    ----------------------
    ``gpt_states`` accepts ``(X, K)`` (interpreted as one source outcome per ``x``)
    or ``(X, A, K)``. ``gpt_effect_set`` is a flat array ``(N_effects, K)`` from
    which valid measurements are inferred as subsets summing to the unit effect.
    Output is a validated ``ContextualityScenario``; with
    ``return_measurement_indices=True`` it also returns the inferred effect-index
    tuples (one tuple per measurement setting ``y``).

    High-level implementation
    -------------------------
    The function: (1) infers measurements from the effect set, (2) computes the
    joint table ``P(a,b|x,y)``, (3) discovers preparation and measurement OPEQs via
    nullspace calculations over GPT vectors, and (4) constructs a
    ``ContextualityScenario`` with those arrays.
    """
    states = np.asarray(gpt_states, dtype=complex)
    if states.ndim == 2:
        states_for_opeq = states[:, np.newaxis, :]
    elif states.ndim == 3:
        states_for_opeq = states
    else:
        raise ValueError("gpt_states must have shape (X, K) or (X, A, K).")

    data_table, measurement_indices = data_table_from_gpt_states_and_effect_set(
        gpt_states=states,
        gpt_effect_set=gpt_effect_set,
        source_outcome_distribution=source_outcome_distribution,
        unit_effect=unit_effect,
        atol=atol,
        outcomes_per_measurement=outcomes_per_measurement,
        drop_tiny_imag=drop_tiny_imag,
    )
    measurement_effects, _ = infer_measurements_from_gpt_effect_set(
        gpt_effect_set=gpt_effect_set,
        unit_effect=unit_effect,
        atol=atol,
        outcomes_per_measurement=outcomes_per_measurement,
    )

    opeq_preps = discover_operational_equivalences_from_gpt_objects(states_for_opeq, atol=atol)
    opeq_meas = discover_operational_equivalences_from_gpt_objects(measurement_effects, atol=atol)

    scenario = ContextualityScenario(
        data=data_table,
        opeq_preps=opeq_preps,
        opeq_meas=opeq_meas,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("\nInferred measurement index sets:")
        for y, idx in enumerate(measurement_indices):
            print(f"y={y}: effects {idx}")
    if return_measurement_indices:
        return scenario, measurement_indices
    return scenario


def contextuality_scenario_from_quantum(
    quantum_states: np.ndarray,
    quantum_effect_set: np.ndarray,
    source_outcome_distribution: np.ndarray | None = None,
    basis: np.ndarray | None = None,
    unit_effect: np.ndarray | None = None,
    atol: float = 1e-9,
    outcomes_per_measurement: int | None = None,
    drop_tiny_imag: bool = True,
    verbose: bool = False,
    return_measurement_indices: bool = False,
) -> ContextualityScenario | tuple[ContextualityScenario, list[tuple[int, ...]]]:
    """Build a ``ContextualityScenario`` from density operators and effects.

    Motivation
    ----------
    Use this when your starting point is quantum objects (matrices), but you want
    the same scenario representation used by the GPT and LP tooling in this package.

    How to use it with other functions
    ----------------------------------
    This is the quantum entry point for downstream randomness analysis. After
    construction, pass the returned scenario to
    ``eve_optimal_guessing_probability`` or
    ``eve_optimal_average_guessing_probability``.

    Input/output structure
    ----------------------
    ``quantum_states`` may be ``(X, d, d)`` or ``(X, A, d, d)``. The effect input is
    a flat set ``quantum_effect_set`` with shape ``(N_effects, d, d)``. Optional
    arguments control basis conversion, tolerance, inferred measurement cardinality,
    and verbosity. Returns a ``ContextualityScenario`` and optionally inferred
    measurement index tuples.

    High-level implementation
    -------------------------
    The function normally converts matrices to GPT vectors in a Hilbert-Schmidt
    Gell-Mann basis, then delegates scenario assembly to
    ``contextuality_scenario_from_gpt``. When states and effects are all projectors
    and no custom basis/unit-effect is supplied, it uses a faster projector
    vectorization path instead. This keeps measurement grouping and OPEQ discovery
    consistent while avoiding unnecessary basis expansion.
    """
    q_states = np.asarray(quantum_states, dtype=complex)
    q_effects = np.asarray(quantum_effect_set, dtype=complex)
    if q_effects.ndim != 3 or q_effects.shape[-2] != q_effects.shape[-1]:
        raise ValueError("quantum_effect_set must have shape (N_effects, d, d).")
    d = q_effects.shape[-1]

    if q_states.ndim == 3:
        if q_states.shape[-2] != q_states.shape[-1]:
            raise ValueError("quantum_states must have square matrices.")
        if q_states.shape[-1] != d:
            raise ValueError("quantum_states and quantum_effect_set dimensions must match.")
    elif q_states.ndim == 4:
        if q_states.shape[-2] != q_states.shape[-1]:
            raise ValueError("quantum_states must have square matrices.")
        if q_states.shape[-1] != d:
            raise ValueError("quantum_states and quantum_effect_set dimensions must match.")
    else:
        raise ValueError("quantum_states must have shape (X,d,d) or (X,A,d,d).")

    use_projector_fast_path = (
        basis is None
        and unit_effect is None
        and _all_projectors(q_states, atol=atol)
        and _all_projectors(q_effects, atol=atol)
    )

    if use_projector_fast_path:
        gpt_states = _matrix_list_to_hs_vectors(q_states, drop_tiny_imag=drop_tiny_imag)
        gpt_effect_set = _matrix_list_to_hs_vectors(q_effects, drop_tiny_imag=drop_tiny_imag)
        unit_effect_for_solver = _matrix_to_hs_vector(np.eye(d, dtype=complex), drop_tiny_imag=drop_tiny_imag)
        if verbose:
            print("Using projector Hilbert-Schmidt vectorization fast path.")
    else:
        gpt_states = convert_matrix_list_to_vector_list(q_states, basis=basis, drop_tiny_imag=drop_tiny_imag)
        gpt_effect_set = convert_matrix_list_to_vector_list(q_effects, basis=basis, drop_tiny_imag=drop_tiny_imag)
        unit_effect_for_solver = unit_effect

    return contextuality_scenario_from_gpt(
        gpt_states=gpt_states,
        gpt_effect_set=gpt_effect_set,
        source_outcome_distribution=source_outcome_distribution,
        unit_effect=unit_effect_for_solver,
        atol=atol,
        outcomes_per_measurement=outcomes_per_measurement,
        drop_tiny_imag=drop_tiny_imag,
        verbose=verbose,
        return_measurement_indices=return_measurement_indices,
    )


# ============================================================================
# Internal helpers
# ============================================================================

def _contains_sympy_entries(obj: object) -> bool:
    try:
        import sympy
    except ImportError:
        return False

    if isinstance(obj, sympy.MatrixBase):
        return True
    if isinstance(obj, sympy.Basic):
        return True
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return any(isinstance(entry, sympy.Basic) for entry in obj.reshape(-1))
    if isinstance(obj, (list, tuple)):
        return any(isinstance(entry, sympy.Basic) for entry in obj)
    return False


def _to_sympy_object_array(values: object) -> np.ndarray:
    import sympy

    arr = np.asarray(values, dtype=object)
    out = np.empty(arr.shape, dtype=object)
    for idx, entry in np.ndenumerate(arr):
        out[idx] = sympy.sympify(entry)
    return out


def _sympy_numeric_complex(value: object) -> complex | None:
    import sympy

    try:
        return complex(sympy.N(value))
    except (TypeError, ValueError):
        return None


def _sympy_numeric_abs(value: object) -> float | None:
    num = _sympy_numeric_complex(value)
    if num is None:
        return None
    return float(abs(num))


def _sympy_real_if_close(value: object, atol: float) -> object:
    import sympy

    expr = sympy.sympify(value)
    imag_part = sympy.simplify(sympy.im(expr))
    if imag_part.is_zero is True:
        return sympy.simplify(sympy.re(expr))
    imag_abs = _sympy_numeric_abs(imag_part)
    if imag_abs is not None and imag_abs <= float(atol):
        return sympy.simplify(sympy.re(expr))
    return expr


def _probability_table_from_gpt_vectors_symbolic(
    gpt_states: np.ndarray,
    gpt_effects: np.ndarray,
    source_outcome_distribution: np.ndarray | None,
    normalize_source_outcomes: bool,
    atol: float,
    drop_tiny_imag: bool,
) -> np.ndarray:
    import sympy

    states = _to_sympy_object_array(gpt_states)
    effects = _to_sympy_object_array(gpt_effects)
    if states.ndim != 3:
        raise ValueError("gpt_states must have shape (X, A, K).")
    if effects.ndim != 3:
        raise ValueError("gpt_effects must have shape (Y, B, K).")
    if states.shape[-1] != effects.shape[-1]:
        raise ValueError("State/effect vector dimensions do not match.")

    probs = np.einsum("xak,ybk->xyab", states, effects)
    if normalize_source_outcomes:
        num_x, _, num_a, _ = probs.shape
        if source_outcome_distribution is None:
            p_a_given_x = np.empty((num_x, num_a), dtype=object)
            p_a_given_x[:, :] = sympy.Rational(1, num_a)
        else:
            p_a_given_x = _to_sympy_object_array(source_outcome_distribution)
            if p_a_given_x.shape != (num_x, num_a):
                raise ValueError(f"source_outcome_distribution must have shape ({num_x}, {num_a}).")
            for x in range(num_x):
                row_sum = sympy.Integer(0)
                for a in range(num_a):
                    value = _sympy_real_if_close(p_a_given_x[x, a], atol=atol)
                    numeric_value = _sympy_numeric_complex(value)
                    if numeric_value is not None:
                        if abs(numeric_value.imag) > float(atol):
                            raise ValueError("source_outcome_distribution contains significant imaginary entries.")
                        if numeric_value.real < -float(atol):
                            raise ValueError("source_outcome_distribution contains negative entries.")
                    p_a_given_x[x, a] = value
                    row_sum += value
                row_gap = sympy.simplify(row_sum - 1)
                row_gap_abs = _sympy_numeric_abs(row_gap)
                if row_gap.is_zero is not True and (row_gap_abs is None or row_gap_abs > float(atol)):
                    raise ValueError("Each source_outcome_distribution[x,:] must sum to 1.")
        probs = probs * p_a_given_x[:, np.newaxis, :, np.newaxis]

    if drop_tiny_imag:
        probs_clean = np.empty(probs.shape, dtype=object)
        for idx, value in np.ndenumerate(probs):
            probs_clean[idx] = _sympy_real_if_close(value, atol=atol)
        return probs_clean
    return probs


def _discover_operational_equivalences_from_gpt_objects_symbolic(
    gpt_objects: np.ndarray,
    atol: float,
) -> np.ndarray:
    import sympy

    raw = _to_sympy_object_array(gpt_objects)
    if raw.ndim == 2:
        raw = raw[np.newaxis, ...]
    if raw.ndim != 3:
        raise ValueError("gpt_objects must have shape (O,K) or (S,O,K).")

    objects = np.empty(raw.shape, dtype=object)
    for idx, value in np.ndenumerate(raw):
        cleaned = _sympy_real_if_close(value, atol=atol)
        imag_part = sympy.simplify(sympy.im(cleaned))
        imag_abs = _sympy_numeric_abs(imag_part)
        if imag_part.is_zero is not True and (imag_abs is None or imag_abs > float(atol)):
            raise ValueError("gpt_objects contains significant imaginary components.")
        objects[idx] = sympy.simplify(sympy.re(cleaned))

    num_s, num_o, vec_dim = objects.shape
    matrix = objects.reshape(num_s * num_o, vec_dim)
    basis = null_space_basis(matrix.T, atol=atol, method="sympy")
    return basis.reshape(-1, num_s, num_o)


def _projector_from_ket_numpy(ket: object, drop_tiny_imag: bool = True) -> np.ndarray:
    vec = np.asarray(ket, dtype=complex)
    if vec.ndim != 1:
        raise ValueError("ket must be a 1D state vector.")
    proj = np.outer(vec, vec.conj())
    return np.real_if_close(proj) if drop_tiny_imag else proj


def _as_sympy_column_vector(ket: object) -> object:
    import sympy

    if isinstance(ket, sympy.MatrixBase):
        vec = ket
    else:
        arr = np.asarray(ket, dtype=object)
        if arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)
        if arr.ndim != 1:
            raise ValueError("ket must be a 1D state vector.")
        vec = sympy.Matrix([sympy.sympify(entry) for entry in arr.tolist()])

    if vec.cols == 1:
        return vec
    if vec.rows == 1:
        return vec.T
    raise ValueError("ket must be a 1D state vector.")


def _projector_from_ket_sympy(ket: object, drop_tiny_imag: bool = True) -> np.ndarray:
    import sympy

    vec = _as_sympy_column_vector(ket)
    proj = vec * vec.conjugate().T
    out = np.empty((proj.rows, proj.cols), dtype=object)
    for i in range(proj.rows):
        for j in range(proj.cols):
            entry = sympy.simplify(proj[i, j])
            if drop_tiny_imag and entry.is_real is True:
                entry = sympy.re(entry)
            out[i, j] = entry
    return out


def _matrix_to_hs_vector_sympy(matrix: object, drop_tiny_imag: bool = True) -> np.ndarray:
    import sympy

    if isinstance(matrix, sympy.MatrixBase):
        mat = matrix
    else:
        arr = np.asarray(matrix, dtype=object)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("matrix must have shape (d, d).")
        mat = sympy.Matrix(
            [[sympy.sympify(arr[i, j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]
        )

    if mat.rows != mat.cols:
        raise ValueError("matrix must have shape (d, d).")

    vec: list[object] = []
    for i in range(mat.rows):
        for j in range(mat.cols):
            entry = sympy.simplify(mat[j, i])  # vec(M^T)
            if drop_tiny_imag and entry.is_real is True:
                entry = sympy.re(entry)
            vec.append(entry)
    return np.asarray(vec, dtype=object)


def _hs_normalize(matrix: np.ndarray) -> np.ndarray:
    norm = np.sqrt(np.trace(matrix @ matrix.conj().T).real)
    if norm <= 0:
        raise ValueError("Cannot normalize zero matrix.")
    return matrix / norm


def _validate_or_build_basis(basis: np.ndarray | None, d: int) -> np.ndarray:
    if basis is None:
        return gell_mann_matrices(d)
    gm = np.asarray(basis, dtype=complex)
    if gm.ndim != 3 or gm.shape != (d * d, d, d):
        raise ValueError(f"basis must have shape ({d*d}, {d}, {d}).")
    return gm


def _matrix_to_hs_vector(matrix: np.ndarray, drop_tiny_imag: bool = True) -> np.ndarray:
    """Vectorize one matrix as ``vec(M^T)`` for trace-compatible dot products."""
    mat = np.asarray(matrix, dtype=complex)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("matrix must have shape (d, d).")
    vec = np.swapaxes(mat, -2, -1).reshape(-1)
    return np.real_if_close(vec) if drop_tiny_imag else vec


def _matrix_list_to_hs_vectors(mats: np.ndarray, drop_tiny_imag: bool = True) -> np.ndarray:
    """Vectorize matrix arrays ``(..., d, d)`` to ``(..., d*d)`` via ``vec(M^T)``."""
    arr = np.asarray(mats, dtype=complex)
    if arr.ndim < 2 or arr.shape[-2] != arr.shape[-1]:
        raise ValueError("mats must have shape (..., d, d).")
    vecs = np.swapaxes(arr, -2, -1).reshape(arr.shape[:-2] + (arr.shape[-1] * arr.shape[-1],))
    return np.real_if_close(vecs) if drop_tiny_imag else vecs


def _all_projectors(mats: np.ndarray, atol: float) -> bool:
    """Check projector conditions ``P^2=P`` and Hermiticity for all matrices."""
    arr = np.asarray(mats, dtype=complex)
    if arr.ndim < 2 or arr.shape[-2] != arr.shape[-1]:
        return False
    idempotent = np.allclose(arr @ arr, arr, atol=atol, rtol=0.0)
    hermitian = np.allclose(arr, np.swapaxes(arr.conj(), -2, -1), atol=atol, rtol=0.0)
    return bool(idempotent and hermitian)

"""Quantum and GPT utilities for contextuality workflows."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .linalg_utils import null_space_basis
from .scenario import ContextualityScenario


# ============================================================================
# Quantum-specific functions
# ============================================================================

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
    drop_tiny_imag: bool = True,
) -> np.ndarray:
    """Compute ``P(a,b|x,y)`` from GPT vectors."""
    states = np.asarray(gpt_states, dtype=complex)
    effects = np.asarray(gpt_effects, dtype=complex)
    if states.ndim != 3:
        raise ValueError("gpt_states must have shape (X, A, K).")
    if effects.ndim != 3:
        raise ValueError("gpt_effects must have shape (Y, B, K).")
    if states.shape[-1] != effects.shape[-1]:
        raise ValueError("State/effect vector dimensions do not match.")

    probs = np.einsum("xak,ybk->xyab", states, effects)
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
    conditional = probability_table_from_gpt_vectors(states, measurement_effects, drop_tiny_imag=drop_tiny_imag)

    num_x, _, num_a, _ = conditional.shape
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

    joint = conditional * p_a_given_x[:, np.newaxis, :, np.newaxis]
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
        data=np.real_if_close(data_table).astype(float),
        opeq_preps=np.real_if_close(opeq_preps).astype(float),
        opeq_meas=np.real_if_close(opeq_meas).astype(float),
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
    ``eve_optimal_average_guessing_probability``. It can also be used through
    ``run_quantum_example`` as a convenience wrapper.

    Input/output structure
    ----------------------
    ``quantum_states`` may be ``(X, d, d)`` or ``(X, A, d, d)``. The effect input is
    a flat set ``quantum_effect_set`` with shape ``(N_effects, d, d)``. Optional
    arguments control basis conversion, tolerance, inferred measurement cardinality,
    and verbosity. Returns a ``ContextualityScenario`` and optionally inferred
    measurement index tuples.

    High-level implementation
    -------------------------
    The function converts matrices to GPT vectors in a Hilbert-Schmidt basis, then
    delegates scenario assembly to ``contextuality_scenario_from_gpt``. This keeps
    all inference steps (measurement grouping, probability-table construction, and
    OPEQ discovery) consistent across GPT and quantum workflows.
    """
    q_states = np.asarray(quantum_states, dtype=complex)
    q_effects = np.asarray(quantum_effect_set, dtype=complex)
    if q_effects.ndim != 3 or q_effects.shape[-2] != q_effects.shape[-1]:
        raise ValueError("quantum_effect_set must have shape (N_effects, d, d).")

    if q_states.ndim == 3:
        gpt_states = convert_matrix_list_to_vector_list(q_states, basis=basis, drop_tiny_imag=drop_tiny_imag)
    elif q_states.ndim == 4:
        gpt_states = convert_matrix_list_to_vector_list(q_states, basis=basis, drop_tiny_imag=drop_tiny_imag)
    else:
        raise ValueError("quantum_states must have shape (X,d,d) or (X,A,d,d).")

    gpt_effect_set = convert_matrix_list_to_vector_list(q_effects, basis=basis, drop_tiny_imag=drop_tiny_imag)
    return contextuality_scenario_from_gpt(
        gpt_states=gpt_states,
        gpt_effect_set=gpt_effect_set,
        source_outcome_distribution=source_outcome_distribution,
        unit_effect=unit_effect,
        atol=atol,
        outcomes_per_measurement=outcomes_per_measurement,
        drop_tiny_imag=drop_tiny_imag,
        verbose=verbose,
        return_measurement_indices=return_measurement_indices,
    )


# ============================================================================
# Internal helpers
# ============================================================================

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

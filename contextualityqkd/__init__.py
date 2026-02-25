"""Public package API for contextuality randomness tooling."""

from .linalg_utils import (
    enumerate_cone_extremal_rays,
    null_space_basis,
    select_linearly_independent_rows,
)
from .extremal_finders import (
    cone_h_to_v_cdd,
    cone_v_to_h_cdd,
    cone_h_to_v_mosek,
    cone_v_to_h_mosek,
)
from .scenario import ContextualityScenario
from .protocol import ContextualityProtocol
from .quantum import (
    GPTContextualityScenario,
    QuantumContextualityScenario,
    convert_matrix_list_to_vector_list,
    convert_matrix_to_vector,
    data_table_from_gpt_states_and_effect_set,
    discover_operational_equivalences_from_gpt_objects,
    discover_operational_equivalences_from_quantum_states,
    direct_probability_table_from_quantum,
    gell_mann_matrices,
    infer_measurements_from_gpt_effect_set,
    maximally_mixed_vector,
    normalize_integer_rays_symbolic,
    probability_table_from_gpt_vectors,
    probability_table_from_quantum_via_gpt,
    group_gpt_vectors_by_indices,
    unit_effect_vector,
    xz_plane_ket,
)


__all__ = [
    "ContextualityScenario",
    "ContextualityProtocol",
    "GPTContextualityScenario",
    "QuantumContextualityScenario",
    "null_space_basis",
    "select_linearly_independent_rows",
    "enumerate_cone_extremal_rays",
    "cone_h_to_v_cdd",
    "cone_v_to_h_cdd",
    "cone_h_to_v_mosek",
    "cone_v_to_h_mosek",
    "convert_matrix_list_to_vector_list",
    "convert_matrix_to_vector",
    "data_table_from_gpt_states_and_effect_set",
    "discover_operational_equivalences_from_gpt_objects",
    "discover_operational_equivalences_from_quantum_states",
    "direct_probability_table_from_quantum",
    "gell_mann_matrices",
    "infer_measurements_from_gpt_effect_set",
    "maximally_mixed_vector",
    "normalize_integer_rays_symbolic",
    "probability_table_from_gpt_vectors",
    "probability_table_from_quantum_via_gpt",
    "group_gpt_vectors_by_indices",
    "unit_effect_vector",
    "xz_plane_ket",
]

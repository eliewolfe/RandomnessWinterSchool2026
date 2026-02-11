"""Public package API for contextuality randomness tooling."""

from .randomness import (
    eve_optimal_average_guessing_probability,
    eve_optimal_guessing_probability,
    min_entropy_bits,
    run_gpt_example,
    run_quantum_example,
)
from .scenario import ContextualityScenario
from .quantum import (
    convert_matrix_list_to_vector_list,
    convert_matrix_to_vector,
    contextuality_scenario_from_gpt,
    contextuality_scenario_from_quantum,
    data_table_from_gpt_states_and_effect_set,
    discover_operational_equivalences_from_gpt_objects,
    discover_operational_equivalences_from_quantum_states,
    direct_probability_table_from_quantum,
    gell_mann_matrices,
    infer_measurements_from_gpt_effect_set,
    maximally_mixed_vector,
    probability_table_from_gpt_vectors,
    probability_table_from_quantum_via_gpt,
    unit_effect_vector,
)


__all__ = [
    "ContextualityScenario",
    "contextuality_scenario_from_gpt",
    "contextuality_scenario_from_quantum",
    "eve_optimal_average_guessing_probability",
    "eve_optimal_guessing_probability",
    "run_quantum_example",
]

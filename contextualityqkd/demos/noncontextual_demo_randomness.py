"""QKD protocol demo: misaligned pairs of XZ states and effects."""
from __future__ import annotations
from pathlib import Path
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import sympy as sp
import numpy as np
from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.quantum import GPTContextualityScenario, QuantumContextualityScenario
from contextualityqkd.scenario import ContextualityScenario

nof_states = 3 # Try switching to 6 to get a super-contextual example!
states = [
    GPTContextualityScenario.projector_hs_vector(
        GPTContextualityScenario.xz_plane_ket(k * 2 * sp.pi / nof_states)
    )    for k in range(nof_states)]


V_states = 1
matrix_id_state = np.array([0.5, 0, 0, 0.5])
states_noise = V_states * np.array(states) + (1 - V_states) * matrix_id_state[np.newaxis, :]

nof_effects = 6
effects = [
    GPTContextualityScenario.projector_hs_vector(
        GPTContextualityScenario.xz_plane_ket(k * 2 * sp.pi / nof_effects)
    )    for k in range(nof_effects)]
measurement_indices = [(2*k % nof_effects,
                        int(2*k+nof_effects/2) % nof_effects) for k in range(nof_effects//2)]
# print("Measurement indices:", measurement_indices)
V_effects = 1
matrix_id_effect = np.array([0.5, 0, 0, 0.5])
quantum_effects_noise = V_effects * np.array(effects) + (1 - V_effects) * matrix_id_effect[np.newaxis, :]

print("\n")
print("Note that effects add up to the unit effect, which is [1,0,0,1].")
print("States:\n", np.array(states, dtype=float))
print("Quantum Effects:\n", np.array(quantum_effects_noise, dtype=float))
# print("Postquantum Effects:\n", np.array(effects, dtype=float))
print("\n")
scenario = GPTContextualityScenario(
    gpt_states=states,
    gpt_effects=quantum_effects_noise,
    measurement_indices=measurement_indices,
    verbose=True)

protocol = ContextualityProtocol(
    scenario,
    where_key=None, # where_key=None → all x for every y
    master_key_holder="Bob")
protocol.print_alice_guessing_metrics()
protocol.print_alice_uncertainty_metrics()
protocol.print_eve_guessing_metrics(method="lp")
protocol.print_eve_uncertainty_metrics(method="lp")
protocol.print_key_rate_summary(method="lp")
scenario.print_contextuality_measures(precision=3)

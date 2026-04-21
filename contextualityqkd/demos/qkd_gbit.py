"""QKD protocol demo: misaligned pairs of XZ states and effects."""

from __future__ import annotations

from pathlib import Path

import sys
import sympy as sp


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.quantum import (
    GPTContextualityScenario, QuantumContextualityScenario,
)
from contextualityqkd.scenario import ContextualityScenario

states = [  GPTContextualityScenario.projector_hs_vector(GPTContextualityScenario.xz_plane_ket(0*sp.pi/2)),
            GPTContextualityScenario.projector_hs_vector(GPTContextualityScenario.xz_plane_ket(1*sp.pi/2)),
            GPTContextualityScenario.projector_hs_vector(GPTContextualityScenario.xz_plane_ket(2*sp.pi/2)),
            GPTContextualityScenario.projector_hs_vector(GPTContextualityScenario.xz_plane_ket(3*sp.pi/2))]
# quantum version
quantum_effects = [ GPTContextualityScenario.projector_hs_vector(GPTContextualityScenario.xz_plane_ket(0*sp.pi/2 + sp.pi/4)),
                    GPTContextualityScenario.projector_hs_vector(GPTContextualityScenario.xz_plane_ket(1*sp.pi/2 + sp.pi/4)),
                    GPTContextualityScenario.projector_hs_vector(GPTContextualityScenario.xz_plane_ket(2*sp.pi/2 + sp.pi/4)),
                    GPTContextualityScenario.projector_hs_vector(GPTContextualityScenario.xz_plane_ket(3*sp.pi/2 + sp.pi/4))]

effects = [ 
[1, 0.5, 0.5, 0],
[0, 0.5, 0.5, 1],
[0, -0.5, -0.5, 1],
[1, -0.5, -0.5, 0]]

print("\n")
print("Note that effects add up to the unit effect, which is [1,0,0,1].")
print("States:\n", np.array(states, dtype=float))
print("Quantum Effects (NOT USED):\n", np.array(quantum_effects, dtype=float))
print("Postquantum Effects:\n", np.array(effects, dtype=float))
print("\n")

scenario = GPTContextualityScenario(
    gpt_states=states,
    gpt_effects=effects,
    measurement_indices=[(0, 2), (1, 3)],
    verbose=True)

# scenario.print_probabilities(as_p_b_given_x_y=True, precision=3, representation="symbolic")
# scenario.print_operational_equivalences(precision=3, representation="symbolic")


protocol = ContextualityProtocol(scenario, where_key=[(0,1,2,3),(0,1,2,3)])

protocol.print_alice_guessing_metrics()
protocol.print_alice_uncertainty_metrics()
protocol.print_eve_guessing_metrics_lp()
protocol.print_eve_uncertainty_metrics_reverse_fano_lp()
protocol.print_key_rate_summary_reverse_fano_lp()

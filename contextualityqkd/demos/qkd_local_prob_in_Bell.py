"""QKD protocol demo: icosahedron-
dodecahedron with Bob-outcome protocol analysis."""

from __future__ import annotations

from pathlib import Path

import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import sympy as sp

from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.quantum import QuantumContextualityScenario
from contextualityqkd.scenario import ContextualityScenario


def build_icosahedron_dodecahedron_scenario(*, eta: float = 1.0) -> QuantumContextualityScenario:
    """Construct Bob-outcome icosahedron-dodecahedron with 20 preparations and 6 binary measurements."""
    eta_f = float(eta)
    if eta_f < 0.0 or eta_f > 1.0:
        raise ValueError("eta must lie in [0,1].")

    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    paulis = [sigma_x, sigma_y, sigma_z]

    quantum_states: list[np.ndarray] = []
    for x0 in (0, 1):
        for x1 in (0, 1):
            for x2 in (0, 1):
                r = np.array([(-1) ** x0, (-1) ** x1, (-1) ** x2], dtype=float) / np.sqrt(3.0)
                rho = 0.5 * (identity + r[0] * sigma_x + r[1] * sigma_y + r[2] * sigma_z)
                rho_eta = eta_f * rho + (1.0 - eta_f) * 0.5 * identity
                quantum_states.append(rho_eta)

    golden_ratio=(1+np.sqrt(5))/2
    for x0 in (0, 1):
        for x1 in (0, 1):
            r = np.array([0, (-1) ** x1 * (1/golden_ratio), (-1) ** x2 * golden_ratio], dtype=float) / np.sqrt(3.0)
            rho = 0.5 * (identity + r[0] * sigma_x + r[1] * sigma_y + r[2] * sigma_z)
            rho_eta = eta_f * rho + (1.0 - eta_f) * 0.5 * identity
            quantum_states.append(rho_eta)
    for x0 in (0, 1):
        for x1 in (0, 1):
            r = np.array([(-1) ** x1 * (1/golden_ratio), (-1) ** x2 * golden_ratio, 0], dtype=float) / np.sqrt(3.0)
            rho = 0.5 * (identity + r[0] * sigma_x + r[1] * sigma_y + r[2] * sigma_z)
            rho_eta = eta_f * rho + (1.0 - eta_f) * 0.5 * identity
            quantum_states.append(rho_eta)
    for x0 in (0, 1):
        for x1 in (0, 1):
            r = np.array([(-1) ** x2 * golden_ratio, 0, (-1) ** x1 * (1/golden_ratio)], dtype=float) / np.sqrt(3.0)
            rho = 0.5 * (identity + r[0] * sigma_x + r[1] * sigma_y + r[2] * sigma_z)
            rho_eta = eta_f * rho + (1.0 - eta_f) * 0.5 * identity
            quantum_states.append(rho_eta)

    N=1/(np.sqrt(1+golden_ratio**2))

    quantum_effects_grouped: list[list[np.ndarray]] = []
    for y in range(2):
        r = np.array([0, N, (-1) ** y * N * golden_ratio], dtype=float)
        plus = 0.5 * (identity + r[0] * sigma_x + r[1] * sigma_y + r[2] * sigma_z)
        minus = 0.5 * (identity - r[0] * sigma_x - r[1] * sigma_y - r[2] * sigma_z)
        quantum_effects_grouped.append([plus, minus])
    for y in range(2):
        r = np.array([(-1) ** y * N * golden_ratio, 0, N], dtype=float)
        plus = 0.5 * (identity + r[0] * sigma_x + r[1] * sigma_y + r[2] * sigma_z)
        minus = 0.5 * (identity - r[0] * sigma_x - r[1] * sigma_y - r[2] * sigma_z)
        quantum_effects_grouped.append([plus, minus])
    for y in range(2):
        r = np.array([N, (-1) ** y * N * golden_ratio, 0], dtype=float)
        plus = 0.5 * (identity + r[0] * sigma_x + r[1] * sigma_y + r[2] * sigma_z)
        minus = 0.5 * (identity - r[0] * sigma_x - r[1] * sigma_y - r[2] * sigma_z)
        quantum_effects_grouped.append([plus, minus])

    return QuantumContextualityScenario.from_quantum_states_effects(
        quantum_states=np.asarray(quantum_states, dtype=complex),
        quantum_effects=np.asarray(quantum_effects_grouped, dtype=complex),
        verbose=False,
    )


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    scenario = build_icosahedron_dodecahedron_scenario(eta=1.0)
    protocol = ContextualityProtocol(scenario, where_key=None)

    ContextualityScenario.print_title("QKD Protocol: icosahedron_dodecahedron (ideal noiseless case)")

    scenario.print_probabilities(as_p_b_given_x_y=True, precision=3, representation="symbolic")

    print("\nOperational equivalences:")
    scenario.print_operational_equivalences(precision=3, representation="symbolic")

    protocol.print_alice_guessing_metrics()
    protocol.print_alice_uncertainty_metrics()
    protocol.print_eve_guessing_metrics_lp()
    protocol.print_eve_uncertainty_metrics_reverse_fano_lp()
    protocol.print_key_rate_summary_reverse_fano_lp()

    auto_protocol = ContextualityProtocol(
        scenario,
        where_key="Automatic",
        optimize_verbose=True,
    )
    auto_protocol.print_where_key_optimization_best_stage(leading_newline=True)

    scenario.print_contextuality_measures(precision=3)


if __name__ == "__main__":
    main()

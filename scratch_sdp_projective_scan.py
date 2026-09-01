"""Scratch scan: does SDP key rate depend on projective toggles for Bob/Eve?

Run with:
    conda run --no-capture-output -n idp python scratch_sdp_projective_scan.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import sympy as sp


_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.quantum import GPTContextualityScenario, QuantumContextualityScenario


@dataclass(frozen=True)
class ScenarioCase:
    name: str
    scenario: object
    where_key: object | None = None
    master_key_holder: str = "Alice"


def _build_porac_scenario(*, eta: float = 1.0) -> QuantumContextualityScenario:
    eta_f = float(eta)
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

    quantum_effects_grouped: list[list[np.ndarray]] = []
    for y in range(3):
        plus = 0.5 * (identity + paulis[y])
        minus = 0.5 * (identity - paulis[y])
        quantum_effects_grouped.append([plus, minus])

    return QuantumContextualityScenario.from_quantum_states_effects(
        quantum_states=np.asarray(quantum_states, dtype=complex),
        quantum_effects=np.asarray(quantum_effects_grouped, dtype=complex),
        verbose=False,
    )


def _build_misaligned_4v6_scenario(
    *,
    state_depolarizing: float = 1.0,
    effect_visibility: float = 1.0,
    effect_offset: object = sp.pi / 10,
) -> GPTContextualityScenario:
    """4 ring preparations with 6 ring effects grouped into 3 binary measurements."""

    state_angles = [k * sp.pi / 2 for k in range(4)]
    effect_angles = [effect_offset + k * 2 * sp.pi / 6 for k in range(6)]

    states = np.array(
        [GPTContextualityScenario.projector_hs_vector(GPTContextualityScenario.xz_plane_ket(theta)) for theta in state_angles],
        dtype=float,
    )
    effects = np.array(
        [GPTContextualityScenario.projector_hs_vector(GPTContextualityScenario.xz_plane_ket(theta)) for theta in effect_angles],
        dtype=float,
    )

    maximally_mixed_state = np.array([0.5, 0.0, 0.0, 0.5], dtype=float)
    unit_effect = np.array([1.0, 0.0, 0.0, 1.0], dtype=float)
    v_s = float(state_depolarizing)
    v_e = float(effect_visibility)
    states = v_s * states + (1.0 - v_s) * maximally_mixed_state[np.newaxis, :]
    effects = v_e * effects + (1.0 - v_e) * 0.5 * unit_effect[np.newaxis, :]

    measurement_indices = ((0, 3), (1, 4), (2, 5))
    return GPTContextualityScenario(
        gpt_states=states,
        gpt_effects=effects,
        measurement_indices=measurement_indices,
        verbose=False,
    )


def _projective_configs() -> list[tuple[bool, bool]]:
    return [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]


def _scan_case(case: ScenarioCase) -> list[dict[str, float | bool | str]]:
    results: list[dict[str, float | bool | str]] = []
    for projective_bob, projective_eve in _projective_configs():
        protocol = ContextualityProtocol(
            scenario=case.scenario,
            where_key=case.where_key,
            master_key_holder=case.master_key_holder,
            atol=1e-9,
            sdp_projective_bob=projective_bob,
            sdp_projective_eve=projective_eve,
            sdp_npa_level_bob=1,
            sdp_npa_level_eve=1,
            sdp_threads=1,
            sdp_verbose=0,
        )

        results.append(
            {
                "scenario": case.name,
                "projective_bob": projective_bob,
                "projective_eve": projective_eve,
                "lp_keyrate": float(protocol.key_rate_per_key_run(method="lp", rate_type="reverse_fano")),
                "sdp_keyrate": float(protocol.key_rate_per_key_run(method="sdp", rate_type="reverse_fano")),
                "lp_guess": float(protocol.eve_guessing_probability(method="lp")),
                "sdp_guess": float(protocol.eve_guessing_probability(method="sdp")),
            }
        )
    return results


def _print_case_results(case_name: str, rows: list[dict[str, float | bool | str]]) -> None:
    print(f"\n=== {case_name} ===")
    print("proj_B proj_E | lp_keyrate sdp_keyrate | lp_guess sdp_guess")
    for row in rows:
        print(
            f"{int(bool(row['projective_bob'])):^6} {int(bool(row['projective_eve'])):^6} | "
            f"{float(row['lp_keyrate']): .6f} {float(row['sdp_keyrate']): .6f} | "
            f"{float(row['lp_guess']): .6f} {float(row['sdp_guess']): .6f}"
        )

    sdp_values = np.array([float(row["sdp_keyrate"]) for row in rows], dtype=float)
    if np.allclose(sdp_values, sdp_values[0], atol=1e-9, rtol=0.0):
        print("Result: SDP key rate is identical across all projective settings.")
    else:
        print("Result: SDP key rate differs across projective settings.")


def main() -> None:
    cases = [
        ScenarioCase(name="PORAC eta=1.0", scenario=_build_porac_scenario(eta=1.0), where_key=None, master_key_holder="Alice"),
        ScenarioCase(name="PORAC eta=0.95", scenario=_build_porac_scenario(eta=0.95), where_key=None, master_key_holder="Alice"),
        ScenarioCase(
            name="Misaligned 4-prep / 3-meas (v_s=0.98, v_e=0.95)",
            scenario=_build_misaligned_4v6_scenario(state_depolarizing=0.98, effect_visibility=0.95, effect_offset=sp.pi / 10),
            where_key=None,
            master_key_holder="Alice",
        ),
        ScenarioCase(
            name="Misaligned 4-prep / 3-meas (v_s=0.90, v_e=0.90)",
            scenario=_build_misaligned_4v6_scenario(state_depolarizing=0.90, effect_visibility=0.90, effect_offset=sp.pi / 7),
            where_key=None,
            master_key_holder="Alice",
        ),
    ]

    all_rows: list[dict[str, float | bool | str]] = []
    for case in cases:
        rows = _scan_case(case)
        all_rows.extend(rows)
        _print_case_results(case.name, rows)

    best_row = max(all_rows, key=lambda item: float(item["sdp_keyrate"]))
    print("\n=== Best observed SDP key rate ===")
    print(
        "scenario={scenario}, projective_bob={projective_bob}, projective_eve={projective_eve}, sdp_keyrate={sdp_keyrate:.6f}".format(
            scenario=best_row["scenario"],
            projective_bob=bool(best_row["projective_bob"]),
            projective_eve=bool(best_row["projective_eve"]),
            sdp_keyrate=float(best_row["sdp_keyrate"]),
        )
    )


if __name__ == "__main__":
    main()


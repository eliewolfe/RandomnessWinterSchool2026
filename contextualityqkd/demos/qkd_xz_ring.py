"""QKD protocol demo: configurable XZ ring with fixed-threshold good-guess where_key."""

from __future__ import annotations

from pathlib import Path

import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.quantum import (
    GPTContextualityScenario,
)
from contextualityqkd.scenario import ContextualityScenario



NUM_MEAS = 6
NUM_STATES = 24
GOOD_GUESS_THRESHOLD = 2.0 / 3.0

def _validate_configuration(num_states: int, num_meas: int) -> None:
    if int(num_states) < 4:
        raise ValueError("NUM_STATES must be at least 4.")
    if int(num_states) % 2 != 0:
        raise ValueError("NUM_STATES must be even so antipodal measurement pairs are well-defined.")
    if int(num_meas) <= 0:
        raise ValueError("NUM_MEAS must be positive.")
    if int(num_meas) > int(num_states) // 2:
        raise ValueError("NUM_MEAS must be <= NUM_STATES // 2 so antipodal pairs remain unique.")


def _build_measurement_indices(num_states: int, num_meas: int) -> tuple[tuple[int, int], ...]:
    """Build approximately evenly spaced antipodal deterministic pairs on the ring."""
    half_ring = int(num_states) // 2
    bases = [(int(m) * half_ring) // int(num_meas) for m in range(int(num_meas))]
    bases = sorted(set(int(idx) for idx in bases))
    if len(bases) != int(num_meas):
        raise RuntimeError(
            "Failed to build unique evenly spaced measurement anchors. "
            "Reduce NUM_MEAS or increase NUM_STATES."
        )
    return tuple((idx, idx + half_ring) for idx in bases)


# def _build_good_guess_where_key(
#     scenario: ContextualityScenario,
#     threshold_alice_guess_bob_probability: float,
# ) -> tuple[tuple[int, ...], ...]:
#     """Include every x for each y where Alice's best Bob guess is above threshold."""
#     threshold = float(threshold_alice_guess_bob_probability)
#     if threshold < 0.0 or threshold > 1.0:
#         raise ValueError("threshold_alice_guess_bob_probability must lie in [0, 1].")

#     guess_xy = np.max(np.asarray(scenario.data_numeric, dtype=float), axis=2)
#     num_x, num_y = guess_xy.shape
#     where_rows: list[tuple[int, ...]] = []
#     for y in range(num_y):
#         keep = np.where(guess_xy[:, y] >= threshold - float(scenario.atol))[0]
#         where_rows.append(tuple(int(x) for x in keep.tolist()))
#     return tuple(where_rows)


# def _realized_guess_threshold(
#     scenario: ContextualityScenario,
#     where_key: tuple[tuple[int, ...], ...],
# ) -> float:
#     """Return the minimum Alice guessing probability over admitted (x,y) pairs."""
#     guess_xy = np.max(np.asarray(scenario.data_numeric, dtype=float), axis=2)
#     admitted: list[float] = []
#     for y, row in enumerate(where_key):
#         for x in row:
#             admitted.append(float(guess_xy[int(x), int(y)]))
#     if not admitted:
#         return float("nan")
#     return float(min(admitted))


def main() -> None:
    num_states = int(NUM_STATES)
    num_meas = int(NUM_MEAS)
    _validate_configuration(num_states, num_meas)

    np.set_printoptions(precision=6, suppress=True)
    ContextualityScenario.print_title(
        f"QKD Protocol: {num_states}-state XZ ring, {num_meas} measurements, "
        "automatic where_key optimization"
    )

    # Evenly spaced 2-outcome measurements from antipodal state/effect pairs.
    measurement_indices = _build_measurement_indices(num_states, num_meas)

    scenario = GPTContextualityScenario.from_xz_ring(
        num_states=num_states,
        measurement_indices=measurement_indices,
        verbose=False,
    )

    scenario.print_probabilities(as_p_b_given_x_y=True, precision=3, representation="numeric")
    scenario.print_operational_equivalences(precision=3, representation="numeric")

    # where_key = _build_good_guess_where_key(
    #     scenario,
    #     threshold_alice_guess_bob_probability=GOOD_GUESS_THRESHOLD,
    # )
    # realized_threshold = _realized_guess_threshold(scenario, where_key)
    # key_counts = np.asarray([len(row) for row in where_key], dtype=int)
    # print("\nFixed where_key from good-guess rule:")
    # print(
    #     "target threshold_alice_guess_bob_probability="
    #     + ContextualityScenario.format_numeric(float(GOOD_GUESS_THRESHOLD), precision=6)
    # )
    # print(
    #     "realized threshold_alice_guess_bob_probability="
    #     + ContextualityScenario.format_numeric(float(realized_threshold), precision=6)
    # )
    # print(
    #     "key counts per y: min="
    #     + str(int(np.min(key_counts)))
    #     + ", max="
    #     + str(int(np.max(key_counts)))
    #     + ", mean="
    #     + ContextualityScenario.format_numeric(float(np.mean(key_counts)), precision=3)
    # )

    protocol = ContextualityProtocol(
        scenario,
        where_key="Automatic",
        optimize_verbose=True,
    )
    protocol.print_where_key_optimization_best_stage(leading_newline=True)
    # protocol.print_alice_guessing_metrics()
    # protocol.print_alice_uncertainty_metrics()
    # protocol.print_eve_guessing_metrics_lp()
    # protocol.print_eve_uncertainty_metrics_reverse_fano_lp()
    protocol.print_key_rate_summary_reverse_fano_lp()

    scenario.print_contextuality_measures(precision=3)


if __name__ == "__main__":
    main()

"""QKD protocol demo: configurable XZ ring with degree-based where_key sweep."""

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


NUM_STATES = 54
NUM_MEAS = 9


def _validate_configuration(num_states: int, num_meas: int) -> None:
    if int(num_states) < 4:
        raise ValueError("NUM_STATES must be at least 4.")
    if int(num_states) % 2 != 0:
        raise ValueError("NUM_STATES must be even so antipodal measurement pairs are well-defined.")
    if int(num_meas) <= 0:
        raise ValueError("NUM_MEAS must be positive.")
    if int(num_states) % (2 * int(num_meas)) != 0:
        raise ValueError("NUM_STATES must be an integer multiple of 2 * NUM_MEAS.")


def _build_measurement_indices(num_states: int, num_meas: int) -> tuple[tuple[int, int], ...]:
    """Build evenly spaced antipodal deterministic pairs on the ring."""
    step = int(num_states) // (2 * int(num_meas))
    return tuple(
        (int(m) * step, int(m) * step + int(num_states) // 2)
        for m in range(int(num_meas))
    )


def _ring_distance(num_states: int, i: int, j: int) -> int:
    delta = (int(i) - int(j)) % int(num_states)
    return int(min(delta, int(num_states) - delta))


def _build_where_key_radius(
    measurement_indices: tuple[tuple[int, int], ...],
    *,
    num_states: int,
    radius: int,
) -> tuple[tuple[int, ...], ...]:
    """Build where_key rows by radius around deterministic endpoint pairs."""
    if radius < 0:
        raise ValueError("radius must be nonnegative.")

    rows: list[tuple[int, ...]] = []
    for state_a, state_b in measurement_indices:
        row = tuple(
            x
            for x in range(num_states)
            if min(
                _ring_distance(num_states, x, state_a),
                _ring_distance(num_states, x, state_b),
            )
            <= radius
        )
        rows.append(row)
    return tuple(rows)


def _build_radius_sweep(num_states: int) -> tuple[tuple[int, int], ...]:
    """Return (radius, expected_count_per_y) entries until all x are key-admissible."""
    sweep: list[tuple[int, int]] = []
    radius = 0
    while True:
        expected = min(num_states, 2 + 4 * radius)
        sweep.append((radius, expected))
        if expected >= num_states:
            break
        radius += 1
    return tuple(sweep)


def _radius_index_to_degrees(num_states: int, radius: int, *, is_final_stage: bool) -> float:
    if is_final_stage:
        return 90.0
    return 360.0 * float(radius) / float(num_states)


def _run_radius_sweep(
    scenario: ContextualityScenario,
    *,
    measurement_indices: tuple[tuple[int, int], ...],
    num_states: int,
) -> None:
    sweep = _build_radius_sweep(num_states)

    stage_stats: list[dict[str, float | int]] = []
    for idx, (radius, expected_count) in enumerate(sweep):
        stage = idx + 1
        is_final_stage = idx == len(sweep) - 1
        radius_deg = _radius_index_to_degrees(num_states, radius, is_final_stage=is_final_stage)

        where_key = _build_where_key_radius(
            measurement_indices,
            num_states=num_states,
            radius=radius,
        )
        protocol = ContextualityProtocol(scenario, where_key=where_key)
        counts = protocol.key_counts_by_y
        if not np.all(counts == expected_count):
            raise RuntimeError(
                f"Radius {radius} expected per-y key count {expected_count}, "
                f"got {counts.tolist()}."
            )

        print(
            "\n"
            + f"Sweep stage {stage}: radius={ContextualityScenario.format_numeric(radius_deg, precision=6)} deg"
        )
        print(f"key counts per y = {counts.tolist()}")
        protocol.print_key_rate_summary_reverse_fano_lp(leading_newline=False)
        stage_stats.append(
            {
                "stage": stage,
                "radius": radius,
                "radius_deg": radius_deg,
                "key_count_per_y": expected_count,
                "key_prob": float(protocol.key_generation_probability_per_run),
                "bits_per_key_run": float(protocol.key_rate_per_key_run_reverse_fano_lp),
                "bits_per_exp_run": float(protocol.key_rate_per_experimental_run_reverse_fano_lp),
            }
        )

    if not stage_stats:
        return

    print("\nSweep summary (reverse Fano):")
    key_probs = [float(stats["key_prob"]) for stats in stage_stats]
    bits_per_key_run = [float(stats["bits_per_key_run"]) for stats in stage_stats]
    bits_per_experimental_run = [float(stats["bits_per_exp_run"]) for stats in stage_stats]
    print(
        "key_probs = "
        + str([ContextualityScenario.format_numeric(value, precision=6) for value in key_probs])
    )
    print(
        "bits_per_key_run = "
        + str([ContextualityScenario.format_numeric(value, precision=6) for value in bits_per_key_run])
    )
    print(
        "bits_per_experimental_run = "
        + str([ContextualityScenario.format_numeric(value, precision=6) for value in bits_per_experimental_run])
    )

    best = max(stage_stats, key=lambda stats: float(stats["bits_per_exp_run"]))
    print(
        "Best protocol by bits per experimental run: "
        f"stage={int(best['stage'])}, "
        f"radius={ContextualityScenario.format_numeric(float(best['radius_deg']), precision=6)} deg, "
        f"bits_per_experimental_run={ContextualityScenario.format_numeric(float(best['bits_per_exp_run']), precision=6)}"
    )


def main() -> None:
    num_states = int(NUM_STATES)
    num_meas = int(NUM_MEAS)
    _validate_configuration(num_states, num_meas)

    np.set_printoptions(precision=6, suppress=True)
    ContextualityScenario.print_title(
        f"QKD Protocol: {num_states}-state XZ ring, {num_meas} measurements, where_key radius sweep"
    )

    # Evenly spaced 2-outcome measurements from antipodal state/effect pairs.
    measurement_indices = _build_measurement_indices(num_states, num_meas)

    scenario = GPTContextualityScenario.from_xz_ring(
        num_states=num_states,
        measurement_indices=measurement_indices,
        verbose=False,
    )

    print("\nNumeric probability table p(b|x,y):")
    scenario.print_probabilities(as_p_b_given_x_y=True, precision=3, representation="numeric")
    scenario.print_operational_equivalences(precision=3, representation="numeric")

    _run_radius_sweep(
        scenario,
        measurement_indices=measurement_indices,
        num_states=num_states,
    )

    scenario.print_contextuality_measures(precision=3)


if __name__ == "__main__":
    main()

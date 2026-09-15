"""QKD demo: no-signalling-only vs full preparation OPEQs.

Six X-Z plane preparations (the four BB84 states plus the two 45-degree states)
and two binary measurements on the CHSH diagonals. The protocol is run twice
on the same behaviour table:

  * with the auto-discovered preparation OPEQs (rank 3), and
  * with only the hand-written pair-mixture "no-signalling" OPEQs (rank 2).

Dropping the extra preparation constraints grants Eve more freedom, so the
second arm yields the weaker key-rate bounds.

Recommended execution:
    python -m contextualityqkd.demos.qkd_NSOPEQs_vs_allOPEQS
"""

from __future__ import annotations

from pathlib import Path

import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import sympy as sp

from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.quantum import (
    GPTContextualityScenario,
)
from contextualityqkd.scenario import ContextualityScenario


# The pair-mixture ("no-signalling") preparation OPEQs, written out by hand:
# each antipodal preparation pair averages to the maximally mixed state, so the
# three mixtures agree. Only two of these are independent (row2 = row0 - row1),
# whereas the auto-discovered preparation OPEQ space has rank 3.
NO_SIGNALLING_PREP_OPEQS = sp.Matrix([
    [-1, -1, 1, 1, 0, 0],   # (rho0+rho1)/2 == (rho2+rho3)/2
    [-1, -1, 0, 0, 1, 1],   # (rho0+rho1)/2 == (rho4+rho5)/2
    [0, 0, 1, 1, -1, -1],   # (rho2+rho3)/2 == (rho4+rho5)/2   [dependent]
])


def build_no_signalling_variant(scenario: ContextualityScenario) -> ContextualityScenario:
    """Same behaviour table, but Eve is only bound by the pair-mixture prep OPEQs.

    Dropping the remaining preparation constraints grants Eve strictly more
    freedom, so every security bound derived from this scenario is weaker.
    """
    return ContextualityScenario(
        data=scenario.data_symbolic,
        opeq_preps=NO_SIGNALLING_PREP_OPEQS,
        opeq_meas=scenario.opeq_meas_symbolic,
        atol=scenario.atol,
        verbose=False,
    )


def _protocol_for(scenario: ContextualityScenario, where_key: object) -> ContextualityProtocol:
    """Build the protocol used by both arms of the comparison."""
    return ContextualityProtocol(
        scenario=scenario,
        where_key=where_key,
        master_key_holder="Alice",
        atol=1e-9,
        lp_solver="highs",
        sdp_solver="MOSEK",
        sdp_projective_bob=False,
        sdp_projective_eve=False,
        sdp_npa_level_bob=1,
        sdp_npa_level_eve=1,
        sdp_use_u_only=True,
        sdp_threads=1,
        sdp_verbose=0,
    )


def compare_opeq_assumptions(
    scenario: ContextualityScenario,
    where_key: object,
    *,
    rate_type: str = "reverse_fano",
) -> None:
    """Run the protocol twice: with the real prep OPEQs, and with only the no-signalling ones."""
    ContextualityScenario.print_title("Comparison: real OPEQs vs no-signalling-only OPEQs")
    ns_scenario = build_no_signalling_variant(scenario)

    arms = [("real OPEQs", scenario), ("no-signalling only", ns_scenario)]
    print(f"\nwhere_key = {where_key}   rate_type = {rate_type}\n")
    header = (
        f"{'assumption':<22}{'prep rank':>10}{'P(keygen)':>11}"
        f"{'LP b/key':>11}{'LP b/run':>11}{'SDP guess':>11}{'SDP b/key':>11}{'SDP b/run':>11}"
    )
    print(header)
    print("-" * len(header))
    for label, sc in arms:
        rank = int(np.linalg.matrix_rank(np.asarray(sc.opeq_preps_numeric, dtype=float)))
        protocol = _protocol_for(sc, where_key)
        lp_key = protocol.key_rate_per_key_run(method="lp", rate_type=rate_type)
        lp_run = protocol.key_rate_per_experimental_run(method="lp", rate_type=rate_type)
        guess = protocol.eve_guess_master_key_sdp
        sdp_key = protocol.key_rate_per_key_run(method="sdp", rate_type=rate_type)
        sdp_run = protocol.key_rate_per_experimental_run(method="sdp", rate_type=rate_type)
        print(
            f"{label:<22}{rank:>10}{protocol.key_generation_probability_per_run:>11.4f}"
            f"{lp_key:>11.6f}{lp_run:>11.6f}{guess:>11.6f}{sdp_key:>11.6f}{sdp_run:>11.6f}"
        )
    print(
        "\nThe no-signalling row drops preparation constraints, so Eve has more freedom "
        "and its bounds are the weaker (lower) ones."
    )


def main() -> None:
    # Keep numerical arrays readable while preserving enough detail.
    np.set_printoptions(precision=6, suppress=True)
    ContextualityScenario.print_title("QKD: no-signalling-only vs full preparation OPEQs")

    # ---------------------------------------------------------------------
    # 1) Define the qubit states/effects in ket form on the X-Z great circle.
    #    - 0 and pi are computational basis |0>,|1> (Z measurement basis)
    #    - +/- pi/2 are |+>,|-> (X basis)
    # ---------------------------------------------------------------------
    # ket0 = GPTContextualityScenario.xz_plane_ket(0)
    # ket1 = GPTContextualityScenario.xz_plane_ket(sp.pi)
    # ket_plus = GPTContextualityScenario.xz_plane_ket(sp.pi / 2)
    # ket_minus = GPTContextualityScenario.xz_plane_ket(-sp.pi / 2)
    
    # state_kets = [ket0, ket1, ket_plus, ket_minus]
    # effect_kets = [ket0, ket1, ket_plus, ket_minus]



    #----------------------------------------------------------------------
    #Another set of preparations and measurement to do the CHSH prepare and measure
    ket0 = GPTContextualityScenario.xz_plane_ket(0)
    ket1 = GPTContextualityScenario.xz_plane_ket(sp.pi)
    ket_plus = GPTContextualityScenario.xz_plane_ket(sp.pi / 2)
    ket_minus = GPTContextualityScenario.xz_plane_ket(-sp.pi / 2)

    e0 = GPTContextualityScenario.xz_plane_ket(sp.pi / 4)
    e1 = GPTContextualityScenario.xz_plane_ket(-3*sp.pi/4)
    e2 = GPTContextualityScenario.xz_plane_ket(3*sp.pi / 4)
    e3 = GPTContextualityScenario.xz_plane_ket(-sp.pi / 4)
    
    state_kets = [ket0, ket1, ket_plus, ket_minus, e0,e1]
    effect_kets = [e0, e1, e2, e3]

    # ---------------------------------------------------------------------
    # 2) Specify preparation and measurement groupings explicitly.
    #    In this QKD-oriented version, preparations are clustered into pairs.
    # ---------------------------------------------------------------------
    preparation_indices = [(0, 1), (2, 3),(4,5)]
    measurement_indices = [(0, 1), (2, 3)]

    # Expose the grouping decisions in the output.
    print("\nProvided preparation index sets:")
    for x, idx in enumerate(preparation_indices):
        print(f"x={x}: preparations {tuple(idx)}")
    print("\nProvided measurement index sets:")
    for y, idx in enumerate(measurement_indices):
        print(f"y={y}: effects {tuple(idx)}")

    # ---------------------------------------------------------------------
    # 3) Convert projectors -> GPT vectors.
    # ---------------------------------------------------------------------
    gpt_state_set = np.array([GPTContextualityScenario.projector_hs_vector(ket) for ket in state_kets], dtype=object)
    gpt_effect_set = np.array([GPTContextualityScenario.projector_hs_vector(ket) for ket in effect_kets], dtype=object)

    # ---------------------------------------------------------------------
    # 4) Build the scenario directly from GPT primitives.
    # ---------------------------------------------------------------------
    scenario = GPTContextualityScenario(
        gpt_states=gpt_state_set,
        gpt_effects=gpt_effect_set,
        measurement_indices=measurement_indices,
        verbose=False,
    )

    # ---------------------------------------------------------------------
    # 5) Print core structural objects and analysis outputs.
    # ---------------------------------------------------------------------
    scenario.print_operational_equivalences(precision=3, representation="symbolic")
    print("\nSymbolic probability table P(a,b|x,y):")
    scenario.print_probabilities(precision=3, representation="symbolic")
    scenario.print_contextuality_measures(metrics=["contextual_fraction"], precision=3, show_inequalities=True, backend_solver="mosek_simplex")

    WHERE_KEY = [(4, 5), ()]

    # Two-arm comparison: real preparation OPEQs vs no-signalling ones only.
    compare_opeq_assumptions(scenario, WHERE_KEY, rate_type="reverse_fano")

    protocol = ContextualityProtocol(
        scenario=scenario,
        where_key=WHERE_KEY,
        master_key_holder="Alice",
        atol=1e-9,
        lp_solver="highs",
        sdp_solver="MOSEK",
        sdp_projective_bob=False,
        sdp_projective_eve=False,
        sdp_npa_level_bob=1,
        sdp_npa_level_eve=1,
        sdp_use_u_only=True,
        sdp_threads=1,
        sdp_verbose=2,
    )
    protocol.print_eve_security_metrics(
        method="both",
        rate_type="reverse_fano",
        include_per_y_lp=False,
        precision_vector=3,
        precision_scalar=6,
        leading_newline=True,
    )
    protocol.print_eve_guess_upper_bound_inequality_by_y()
    protocol.print_eve_guess_upper_bound_inequality()

if __name__ == "__main__":
    main()

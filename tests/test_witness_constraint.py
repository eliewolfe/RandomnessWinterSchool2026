"""Tests for the witness-only data-constraint mode (LP and SDP).

In witness mode Eve is constrained by the operational equivalences plus a
single linear inequality on her Bob-marginal, instead of full consistency with
the observed behavior. This is always a relaxation, so her guessing
probability can only go up and the certified key rate can only go down.
"""

from __future__ import annotations

import unittest

import numpy as np
import sympy as sp

from contextualityqkd.contextuality import NoncontextualityAssessment
from contextualityqkd.eve_lp import QKDNoncontextualLP
from contextualityqkd.eve_sdp import QKDNoncontextualSDP
from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.quantum import GPTContextualityScenario


def _xz_misaligned_scenario() -> GPTContextualityScenario:
    G = GPTContextualityScenario
    states = [G.projector_hs_vector(G.xz_plane_ket(k * sp.pi / 2)) for k in range(4)]
    effects = [G.projector_hs_vector(G.xz_plane_ket(k * sp.pi / 2 + sp.pi / 4)) for k in range(4)]
    return GPTContextualityScenario(
        gpt_states=states,
        gpt_effects=effects,
        measurement_indices=[(0, 2), (1, 3)],
        verbose=False,
    )


class TestWitnessConstraintMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenario = _xz_misaligned_scenario()
        cls.where_key = [(0, 1, 2, 3), ()]
        cls.protocol = ContextualityProtocol(
            scenario=cls.scenario, where_key=cls.where_key,
            master_key_holder="Alice", lp_solver="highs",
        )
        cls.full_guess = float(cls.protocol.eve_guess_master_key_average_y_lp)

    def test_constructor_validation(self) -> None:
        with self.assertRaises(ValueError):
            QKDNoncontextualLP(self.scenario, data_constraint="witness")
        with self.assertRaises(ValueError):
            QKDNoncontextualLP(
                self.scenario, witness_coeffs=np.zeros((4, 2, 2)), witness_bound=0.5
            )

    def test_guessing_dual_witness_reproduces_full_guess(self) -> None:
        # The refined per-y guessing-bound dual says G <= <c, P>; imposing
        # <c, P> <= w_obs (its value at the data) must reproduce G exactly.
        solver = self.protocol.eve_master_key_lp_solver
        coeffs = solver.guess_bound_coeffs_by_y[0]
        w_obs = float(np.sum(coeffs * np.asarray(self.scenario.data_numeric)))
        result = self.protocol.key_rate_from_witness(
            coeffs, w_obs, witness_sense="<=", method="lp"
        )
        self.assertAlmostEqual(result["eve_guess"], self.full_guess, places=6)
        self.assertAlmostEqual(
            result["key_rate_per_key_run"],
            self.protocol.key_rate_per_key_run(method="lp"),
            places=6,
        )

    def test_contextuality_witness_mode_is_sound_relaxation(self) -> None:
        assessment = NoncontextualityAssessment(
            self.scenario, monotone="contextual_fraction", backend_solver="highs",
            qp_solver="clarabel",
        )
        alpha = assessment.inequality["contextual_fraction"]
        sense = assessment.inequality_sense["contextual_fraction"]
        observed = float(np.sum(alpha * np.asarray(self.scenario.data_numeric)))
        # Constrain Eve by the observed witness value in the violating direction.
        witness_sense = ">=" if sense == "<=" else "<="
        result = self.protocol.key_rate_from_witness(
            alpha, observed, witness_sense=witness_sense, method="lp"
        )
        self.assertGreaterEqual(result["eve_guess"], self.full_guess - 1e-8)
        self.assertLessEqual(result["eve_guess"], 1.0 + 1e-8)
        self.assertLessEqual(
            result["key_rate_per_key_run"],
            self.protocol.key_rate_per_key_run(method="lp") + 1e-8,
        )

    def test_witness_mode_monotone_in_bound(self) -> None:
        solver = self.protocol.eve_master_key_lp_solver
        coeffs = solver.guess_bound_coeffs_by_y[0]
        w_obs = float(np.sum(coeffs * np.asarray(self.scenario.data_numeric)))
        loose = self.protocol.key_rate_from_witness(
            coeffs, w_obs + 0.05, witness_sense="<=", method="lp"
        )
        tight = self.protocol.key_rate_from_witness(
            coeffs, w_obs, witness_sense="<=", method="lp"
        )
        self.assertGreaterEqual(loose["eve_guess"], tight["eve_guess"] - 1e-8)

    def test_sdp_witness_mode_runs_and_relaxes(self) -> None:
        solver_lp = self.protocol.eve_master_key_lp_solver
        coeffs = solver_lp.guess_bound_coeffs_by_y[0]
        w_obs = float(np.sum(coeffs * np.asarray(self.scenario.data_numeric)))
        sdp = QKDNoncontextualSDP(
            self.scenario, npa_level_bob=1, npa_level_eve=1, use_u_only=True,
            master_key_holder="Alice", where_key=self.where_key, threads=1,
            data_constraint="witness", witness_coeffs=coeffs,
            witness_bound=w_obs, witness_sense="<=",
        )
        sdp.solve_sdp()
        witness_guess = float(sdp.eve_success_probability)
        sdp_full = QKDNoncontextualSDP(
            self.scenario, npa_level_bob=1, npa_level_eve=1, use_u_only=True,
            master_key_holder="Alice", where_key=self.where_key, threads=1,
        )
        sdp_full.solve_sdp()
        full_guess = float(sdp_full.eve_success_probability)
        self.assertGreaterEqual(witness_guess, full_guess - 1e-6)
        self.assertLessEqual(witness_guess, w_obs + 1e-6)


if __name__ == "__main__":
    unittest.main()

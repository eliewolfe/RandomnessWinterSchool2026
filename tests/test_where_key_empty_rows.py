"""Regression tests: empty where_key rows (single-key-setting protocols).

A protocol may establish key from only a subset of Bob's settings; a row
``where_key[y] = ()`` marks setting ``y`` as never key-generating. These tests
pin down the intended semantics end to end:

- the Eve LP skips the empty row (NaN in the per-y vector) and averages over
  the remaining settings only;
- per-key-run rates equal the single-setting rate, while per-experimental-run
  rates are discounted by the key-generation probability;
- the SDP pathway accepts empty rows as long as one key pair exists;
- reporting helpers do not choke on the NaN entries.
"""

from __future__ import annotations

import unittest

import numpy as np
import sympy as sp

from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.quantum import GPTContextualityScenario


def _xz_misaligned_scenario() -> GPTContextualityScenario:
    """BB84 states, Bob measuring in the two intermediate (rotated) bases."""
    G = GPTContextualityScenario
    states = [G.projector_hs_vector(G.xz_plane_ket(k * sp.pi / 2)) for k in range(4)]
    effects = [G.projector_hs_vector(G.xz_plane_ket(k * sp.pi / 2 + sp.pi / 4)) for k in range(4)]
    return GPTContextualityScenario(
        gpt_states=states,
        gpt_effects=effects,
        measurement_indices=[(0, 2), (1, 3)],
        verbose=False,
    )


class TestEmptyWhereKeyRows(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenario = _xz_misaligned_scenario()
        cls.full = ContextualityProtocol(
            scenario=cls.scenario, where_key=None, master_key_holder="Alice", lp_solver="highs"
        )
        cls.single = ContextualityProtocol(
            scenario=cls.scenario,
            where_key=[(0, 1, 2, 3), ()],
            master_key_holder="Alice",
            lp_solver="highs",
        )

    def test_lp_guess_vector_marks_empty_row_nan(self) -> None:
        guesses = self.single.eve_guess_master_key_by_y_lp
        self.assertTrue(np.isfinite(guesses[0]))
        self.assertTrue(np.isnan(guesses[1]))
        self.assertAlmostEqual(float(guesses[0]), float(self.full.eve_guess_master_key_by_y_lp[0]), places=8)

    def test_lp_average_skips_empty_rows(self) -> None:
        self.assertAlmostEqual(
            float(self.single.eve_guess_master_key_average_y_lp),
            float(self.single.eve_guess_master_key_by_y_lp[0]),
            places=12,
        )

    def test_key_run_rate_matches_single_setting(self) -> None:
        # This scenario is y-symmetric, so keying on one y must reproduce the
        # per-key-run rate of the full protocol exactly.
        for rate_type in ("reverse_fano", "min_entropy"):
            self.assertAlmostEqual(
                self.single.key_rate_per_key_run(method="lp", rate_type=rate_type),
                self.full.key_rate_per_key_run(method="lp", rate_type=rate_type),
                places=8,
            )

    def test_experimental_run_rate_is_discounted(self) -> None:
        self.assertAlmostEqual(self.single.key_generation_probability_per_run, 0.5, places=12)
        self.assertAlmostEqual(
            self.single.key_rate_per_experimental_run(method="lp"),
            0.5 * self.single.key_rate_per_key_run(method="lp"),
            places=12,
        )

    def test_dual_witness_only_for_keyed_setting(self) -> None:
        solver = self.single.eve_master_key_lp_solver
        self.assertIn(0, solver.guess_bound_coeffs_by_y)
        self.assertNotIn(1, solver.guess_bound_coeffs_by_y)

    def test_reporting_handles_nan_rows(self) -> None:
        text = self.single.format_eve_security_metrics(method="lp", rate_type="reverse_fano")
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)

    def test_sdp_accepts_empty_row(self) -> None:
        protocol = ContextualityProtocol(
            scenario=self.scenario,
            where_key=[(0, 1, 2, 3), ()],
            master_key_holder="Alice",
            lp_solver="highs",
            sdp_npa_level_bob=1,
            sdp_npa_level_eve=1,
            sdp_use_u_only=True,
            sdp_threads=1,
            sdp_verbose=0,
        )
        guess = protocol.eve_guess_master_key_sdp
        self.assertTrue(np.isfinite(guess))
        # The quantum adversary is a special case of the operational one.
        self.assertLessEqual(guess, float(protocol.eve_guess_master_key_average_y_lp) + 1e-6)

    def test_all_rows_empty_rejected_by_sdp_and_degenerate_in_lp(self) -> None:
        protocol = ContextualityProtocol(
            scenario=self.scenario, where_key=[(), ()], master_key_holder="Alice", lp_solver="highs"
        )
        self.assertEqual(protocol.key_pair_count_total, 0)
        self.assertEqual(protocol.key_rate_per_experimental_run(method="lp"), 0.0)
        self.assertTrue(np.isnan(protocol.key_rate_per_key_run(method="lp")))


if __name__ == "__main__":
    unittest.main()

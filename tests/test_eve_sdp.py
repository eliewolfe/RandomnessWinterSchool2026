import unittest

import numpy as np

from contextualityqkd.eve_sdp import QKDNoncontextualSDP, _shannon_entropy
from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.scenario import ContextualityScenario


class EveSDPTests(unittest.TestCase):
    def test_builds_first_level_template(self) -> None:
        data = np.array([[[0.7, 0.3], [0.4, 0.6]]], dtype=float)
        scenario = ContextualityScenario(data)
        solver = QKDNoncontextualSDP(scenario, npa_level_bob=1, npa_level_eve=1)

        operators = solver.build_operator_list()
        solver.build_lexorder_and_notcomm()
        matrices = solver.instantiate_moment_matrices()

        self.assertEqual(len(operators), 8)
        self.assertEqual(len(matrices), scenario.X_cardinality)
        self.assertIsNotNone(solver.template)
        self.assertGreater(solver.template.dimension, 1)

    def test_solves_toy_eve_success_bound(self) -> None:
        data = np.array([[[0.8, 0.2]]], dtype=float)
        scenario = ContextualityScenario(data)
        solver = QKDNoncontextualSDP(scenario, npa_level_bob=1, npa_level_eve=1, threads=1)

        value = solver.solve_sdp()

        self.assertTrue(np.isfinite(value))
        self.assertGreaterEqual(value, -1e-7)
        self.assertLessEqual(value, 1.0 + 1e-6)
        self.assertTrue(np.isfinite(solver.key_rate_lower_bound))

    def test_shannon_entropy_helper(self) -> None:
        self.assertAlmostEqual(_shannon_entropy([0.5, 0.5]), 1.0)

    def test_protocol_lp_bounds_sdp_for_both_master_keys(self) -> None:
        data = np.array(
            [
                [[0.8, 0.2], [0.55, 0.45]],
                [[0.3, 0.7], [0.4, 0.6]],
            ],
            dtype=float,
        )
        scenario = ContextualityScenario(data)
        where_key = [(0, 1), (0, 1)]

        for holder in ("Alice", "Bob"):
            protocol = ContextualityProtocol(
                scenario,
                where_key=where_key,
                master_key_holder=holder,
                sdp_threads=1,
            )
            self.assertGreaterEqual(
                protocol.eve_guessing_probability(method="lp") + 1e-6,
                protocol.eve_guessing_probability(method="sdp"),
            )
            for rate_type in ("reverse_fano", "min_entropy"):
                self.assertGreaterEqual(
                    protocol.key_rate_per_key_run(method="sdp", rate_type=rate_type) + 1e-6,
                    protocol.key_rate_per_key_run(method="lp", rate_type=rate_type),
                )

    def test_cached_both_method_reporting_blocks(self) -> None:
        data = np.array([[[0.8, 0.2], [0.55, 0.45]]], dtype=float)
        scenario = ContextualityScenario(data)
        protocol = ContextualityProtocol(scenario, where_key=[(0,), (0,)], sdp_threads=1)

        metrics = protocol.eve_guessing_metrics(method="both")
        self.assertIs(metrics, protocol.eve_guessing_metrics(method="both"))
        self.assertEqual([block["method"] for block in metrics], ["lp", "sdp"])

        formatted = protocol.format_eve_security_metrics(method="both", rate_type="reverse_fano")
        self.assertLess(formatted.index("Eve LP guessing metrics"), formatted.index("Eve SDP guessing metrics"))
        self.assertIn("Key-rate summary (reverse Fano, LP", formatted)
        self.assertIn("Key-rate summary (reverse Fano, SDP", formatted)


if __name__ == "__main__":
    unittest.main()

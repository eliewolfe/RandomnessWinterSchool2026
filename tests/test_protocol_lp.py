import unittest

import numpy as np

from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.scenario import ContextualityScenario


class ProtocolLPTests(unittest.TestCase):
    def test_eve_lp_vector_and_average(self) -> None:
        data = np.array(
            [
                [[0.8, 0.2], [0.55, 0.45]],
                [[0.3, 0.7], [0.4, 0.6]],
            ],
            dtype=float,
        )
        scenario = ContextualityScenario(data)
        protocol = ContextualityProtocol(scenario, where_key=[[], [0, 1]], master_key_holder="Bob")

        vec = protocol.eve_guess_master_key_by_y_lp
        self.assertEqual(vec.shape, (2,))
        self.assertTrue(np.isnan(vec[0]))
        self.assertTrue(np.isfinite(vec[1]))
        self.assertGreaterEqual(vec[1], 0.0)
        self.assertLessEqual(vec[1], 1.0)
        self.assertAlmostEqual(protocol.eve_guess_master_key_average_y_lp, float(vec[1]))

    def test_reverse_fano_keyrate_outputs(self) -> None:
        data = np.array(
            [
                [[0.75, 0.25], [0.5, 0.5]],
                [[0.35, 0.65], [0.45, 0.55]],
            ],
            dtype=float,
        )
        scenario = ContextualityScenario(data)
        protocol = ContextualityProtocol(scenario, master_key_holder="Alice")

        self.assertTrue(np.all(np.isfinite(protocol.eve_uncertainty_master_key_reverse_fano_by_y_lp)))
        self.assertTrue(np.isfinite(protocol.key_rate_per_key_run_reverse_fano_lp))
        self.assertTrue(np.isfinite(protocol.key_rate_per_experimental_run_reverse_fano_lp))
        np.testing.assert_allclose(
            protocol.key_rate_by_y_reverse_fano_lp,
            protocol.eve_uncertainty_master_key_reverse_fano_by_y_lp - protocol.other_party_uncertainty_by_y,
        )

    def test_lp_reporting_defaults_to_averages_with_optional_per_y_detail(self) -> None:
        data = np.array(
            [
                [[0.8, 0.2], [0.55, 0.45]],
                [[0.3, 0.7], [0.4, 0.6]],
            ],
            dtype=float,
        )
        scenario = ContextualityScenario(data)
        protocol = ContextualityProtocol(scenario, master_key_holder="Alice")

        guess_text = protocol.format_eve_guessing_metrics(method="lp")
        self.assertIn("P_E^guess(master_key|Y) (LP)", guess_text)
        self.assertNotIn("P_E^guess(master_key|y) (LP)", guess_text)

        guess_text_detailed = protocol.format_eve_guessing_metrics(method="lp", include_per_y_lp=True)
        self.assertIn("P_E^guess(master_key|y) (LP)", guess_text_detailed)

        uncertainty_text = protocol.format_eve_uncertainty_metrics(method="lp", rate_type="reverse_fano")
        self.assertIn("H_E(master_key|Y)", uncertainty_text)
        self.assertNotIn("H_E(master_key|y)", uncertainty_text)

        uncertainty_text_detailed = protocol.format_eve_uncertainty_metrics(
            method="lp",
            rate_type="reverse_fano",
            include_per_y_lp=True,
        )
        self.assertIn("H_E(master_key|y)", uncertainty_text_detailed)

        summary_text = protocol.format_key_rate_summary(method="lp", rate_type="reverse_fano")
        self.assertNotIn("bits per key-generating run by y", summary_text)

        summary_text_detailed = protocol.format_key_rate_summary(
            method="lp",
            rate_type="reverse_fano",
            include_per_y_lp=True,
        )
        self.assertIn("bits per key-generating run by y (LP)", summary_text_detailed)


if __name__ == "__main__":
    unittest.main()

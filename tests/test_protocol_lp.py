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
        protocol = ContextualityProtocol(scenario, where_key=[[], [0, 1]])

        vec = protocol.eve_guess_bob_by_y_lp
        self.assertEqual(vec.shape, (2,))
        self.assertTrue(np.isnan(vec[0]))
        self.assertTrue(np.isfinite(vec[1]))
        self.assertGreaterEqual(vec[1], 0.0)
        self.assertLessEqual(vec[1], 1.0)
        self.assertAlmostEqual(protocol.eve_guess_bob_average_y_lp, float(vec[1]))

    def test_reverse_fano_keyrate_outputs(self) -> None:
        data = np.array(
            [
                [[0.75, 0.25], [0.5, 0.5]],
                [[0.35, 0.65], [0.45, 0.55]],
            ],
            dtype=float,
        )
        scenario = ContextualityScenario(data)
        protocol = ContextualityProtocol(scenario)

        self.assertTrue(np.all(np.isfinite(protocol.eve_uncertainty_bob_reverse_fano_by_y_lp)))
        self.assertTrue(np.isfinite(protocol.key_rate_per_key_run_reverse_fano_lp))
        self.assertTrue(np.isfinite(protocol.key_rate_per_experimental_run_reverse_fano_lp))


if __name__ == "__main__":
    unittest.main()

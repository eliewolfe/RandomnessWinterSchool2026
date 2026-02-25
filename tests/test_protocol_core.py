import unittest

import numpy as np

from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.scenario import ContextualityScenario


class ProtocolCoreTests(unittest.TestCase):
    def _build_small_scenario(self) -> ContextualityScenario:
        data = np.array(
            [
                [[0.9, 0.1], [0.6, 0.4]],
                [[0.2, 0.8], [0.7, 0.3]],
            ],
            dtype=float,
        )
        return ContextualityScenario(data)

    def test_where_key_normalization(self) -> None:
        protocol = ContextualityProtocol(self._build_small_scenario(), where_key=[[1, 0, 1], [1]])
        self.assertEqual(protocol.where_key, ((0, 1), (1,)))

    def test_alice_guess_tables(self) -> None:
        protocol = ContextualityProtocol(self._build_small_scenario(), where_key=[[0, 1], [1]])

        expected_xy = np.array(
            [
                [0.9, 0.6],
                [0.8, 0.7],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(protocol.alice_guess_bob_by_xy, expected_xy)
        np.testing.assert_allclose(protocol.alice_guess_bob_by_y_key, np.array([0.85, 0.7]))
        self.assertAlmostEqual(protocol.alice_guess_bob_key_weighted, 0.8)

    def test_empty_row_nan_behavior(self) -> None:
        protocol = ContextualityProtocol(self._build_small_scenario(), where_key=[[], [0]])
        self.assertTrue(np.isnan(protocol.alice_guess_bob_by_y_key[0]))
        self.assertTrue(np.isnan(protocol.alice_uncertainty_bob_by_y_key[0]))
        self.assertEqual(protocol.key_pair_count_total, 1)
        self.assertAlmostEqual(protocol.key_generation_probability_per_run, 0.25)


if __name__ == "__main__":
    unittest.main()

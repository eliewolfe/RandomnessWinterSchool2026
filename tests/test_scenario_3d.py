import unittest

import numpy as np

from contextualityqkd.scenario import ContextualityScenario


class Scenario3DTests(unittest.TestCase):
    def test_dense_behavior_table_normalization(self) -> None:
        data = np.array(
            [
                [[0.7, 0.3], [0.6, 0.4]],
                [[0.2, 0.8], [0.5, 0.5]],
            ],
            dtype=float,
        )
        scenario = ContextualityScenario(data)
        self.assertEqual(scenario.data_numeric.shape, (2, 2, 2))
        np.testing.assert_allclose(scenario.b_cardinality_per_y, np.array([2, 2]))
        scenario.sanity_check()

    def test_ragged_b_per_y_behavior(self) -> None:
        data_ragged = [
            [[0.7, 0.3], [0.2, 0.3, 0.5]],
            [[0.4, 0.6], [0.1, 0.4, 0.5]],
        ]
        scenario = ContextualityScenario(data_ragged)
        self.assertEqual(scenario.data_numeric.shape, (2, 2, 3))
        np.testing.assert_array_equal(scenario.b_cardinality_per_y, np.array([2, 3]))
        np.testing.assert_allclose(scenario.data_numeric[:, 0, 2], 0.0)
        scenario.sanity_check()

    def test_prep_opeq_shape_accepts_vector(self) -> None:
        data = np.array(
            [
                [[0.5, 0.5], [0.4, 0.6]],
                [[0.5, 0.5], [0.4, 0.6]],
            ],
            dtype=float,
        )
        scenario = ContextualityScenario(data, opeq_preps=np.array([1.0, -1.0]))
        self.assertEqual(scenario.opeq_preps_numeric.shape[1], 2)


if __name__ == "__main__":
    unittest.main()

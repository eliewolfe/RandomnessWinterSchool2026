import unittest
import warnings
from unittest.mock import patch

import numpy as np

from contextualityqkd.linalg_utils import enumerate_cone_extremal_rays


class ConeEnumerationTests(unittest.TestCase):
    def test_cdd_empty_rays_falls_back_to_mosek(self) -> None:
        equalities = np.array([[1.0, -1.0, 0.0]], dtype=float)
        fallback_rays = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        empty = np.empty((0, 3), dtype=float)

        with patch(
            "contextualityqkd.extremal_finders.cone_h_to_v_cdd",
            return_value=(empty, empty),
        ) as mock_cdd:
            with patch(
                "contextualityqkd.extremal_finders.cone_h_to_v_mosek",
                return_value=(fallback_rays, empty),
            ) as mock_mosek:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    rays = enumerate_cone_extremal_rays(equalities, method="cdd")

        np.testing.assert_allclose(rays, fallback_rays)
        self.assertEqual(mock_cdd.call_count, 1)
        self.assertEqual(mock_mosek.call_count, 1)
        self.assertTrue(any("MOSEK fallback" in str(entry.message) for entry in caught))

    def test_cdd_and_mosek_empty_rays_raise(self) -> None:
        equalities = np.array([[1.0, -1.0, 0.0]], dtype=float)
        empty = np.empty((0, 3), dtype=float)

        with patch(
            "contextualityqkd.extremal_finders.cone_h_to_v_cdd",
            return_value=(empty, empty),
        ):
            with patch(
                "contextualityqkd.extremal_finders.cone_h_to_v_mosek",
                return_value=(empty, empty),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "CDD and MOSEK backends",
                ):
                    enumerate_cone_extremal_rays(equalities, method="cdd")


if __name__ == "__main__":
    unittest.main()

"""Tests for ContextualityScenario.restricted_to_completeness_meas_opeqs.

The derived scenario keeps the behavior and preparation OPEQs but forgets every
measurement operational equivalence beyond the unit measurement trace (the
prepare-and-measure analogue of no-signalling). Eve can only get stronger.
"""

from __future__ import annotations

import unittest

import numpy as np
import sympy as sp

from contextualityqkd.eve_lp import QKDNoncontextualLP
from contextualityqkd.quantum import GPTContextualityScenario
from contextualityqkd.scenario import ContextualityScenario


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


class TestCompletenessOnlyMeasOpeqs(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.full = _xz_misaligned_scenario()
        cls.restricted = cls.full.restricted_to_completeness_meas_opeqs()

    def test_data_and_prep_opeqs_survive(self) -> None:
        np.testing.assert_allclose(self.restricted.data_numeric, self.full.data_numeric, atol=1e-12)
        np.testing.assert_allclose(
            self.restricted.opeq_preps_numeric, self.full.opeq_preps_numeric, atol=1e-12
        )
        np.testing.assert_array_equal(
            self.restricted.b_cardinality_per_y, self.full.b_cardinality_per_y
        )

    def test_completeness_rows_annihilate_data(self) -> None:
        rows = self.full.completeness_opeq_meas
        residual = np.tensordot(rows, self.full.data_numeric, axes=([1, 2], [1, 2]))
        np.testing.assert_allclose(residual, 0.0, atol=1e-9)

    def test_restriction_is_a_relaxation_for_eve(self) -> None:
        guess_full = QKDNoncontextualLP(self.full, backend_solver="highs").solve_lp()
        guess_restricted = QKDNoncontextualLP(self.restricted, backend_solver="highs").solve_lp()
        self.assertTrue(np.all(guess_restricted >= guess_full - 1e-8))

    def test_fixed_point_when_only_completeness_exists(self) -> None:
        # A generic (contextuality-free) 2x2x2 behavior: the only measurement
        # OPEQs the nullspace can contain are trace-type combinations, so the
        # restricted scenario must give Eve exactly the same LP power.
        rng = np.random.default_rng(7)
        raw = rng.random((2, 2, 2))
        data = raw / raw.sum(axis=2, keepdims=True)
        scenario = ContextualityScenario(data=data, atol=1e-9, verbose=False)
        restricted = scenario.restricted_to_completeness_meas_opeqs()
        g1 = QKDNoncontextualLP(scenario, backend_solver="highs").solve_lp()
        g2 = QKDNoncontextualLP(restricted, backend_solver="highs").solve_lp()
        np.testing.assert_allclose(g1, g2, atol=1e-7)


if __name__ == "__main__":
    unittest.main()

import unittest

import cvxpy as cp
import numpy as np

from contextualityqkd.contextuality import NoncontextualityAssessment, _default_dephasing_target
from contextualityqkd.quantum import GPTContextualityScenario
from contextualityqkd.scenario import ContextualityScenario


def _cabello_scenario() -> GPTContextualityScenario:
    labels = list("123456789ABCDEFGHI")
    rays = np.array(
        [
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 1, 1, 1], [1, -1, 1, -1],
            [1, -1, -1, 1], [1, -1, -1, -1], [1, -1, 1, 1], [1, 1, 1, -1], [1, 1, 0, 0],
            [0, 0, 1, 1], [0, 0, 1, -1], [0, 1, 0, 1], [0, 1, 0, -1], [1, 0, -1, 0],
            [1, 0, 0, -1], [1, 0, 0, 1], [0, 1, -1, 0],
        ],
        dtype=int,
    )
    contexts = ["12BC", "13DE", "23GH", "45EF", "46GI", "56AB", "78AC", "79HI", "89DF"]
    lab = {l: i for i, l in enumerate(labels)}
    mi = [tuple(lab[c] for c in ctx) for ctx in contexts]
    return GPTContextualityScenario.from_integer_rays(rays=rays, measurement_indices=mi, verbose=False)


def _noncontextual_222() -> ContextualityScenario:
    data = np.array([[[0.8, 0.2], [0.55, 0.45]], [[0.3, 0.7], [0.4, 0.6]]], dtype=float)
    return ContextualityScenario(data)


def _noncontextual_extremal_products(assessment: NoncontextualityAssessment) -> np.ndarray:
    """All noncontextual extremal ray products, flattened over (x, y, b) in C order."""
    prep = assessment.prep_extremals
    effect_flat = assessment.effect_extremals.reshape(assessment.effect_extremals.shape[0], -1)
    n = assessment.num_x * assessment.num_y * assessment.num_b
    return np.einsum("ix,jq->ijxq", prep, effect_flat).reshape(-1, n)


class ContextualityParityTests(unittest.TestCase):
    def test_cabello_is_maximally_contextual(self) -> None:
        a = NoncontextualityAssessment(_cabello_scenario(), monotone="both")
        self.assertAlmostEqual(a.contextual_fraction, 1.0, places=6)
        self.assertAlmostEqual(a.noncontextual_fraction, 0.0, places=6)
        self.assertAlmostEqual(a.dephasing_robustness, 1.0 / 3.0, places=6)
        self.assertTrue(a.contextual)
        self.assertFalse(a.is_simplex_embeddable)

    def test_noncontextual_scenario(self) -> None:
        a = NoncontextualityAssessment(_noncontextual_222(), monotone="both")
        self.assertAlmostEqual(a.contextual_fraction, 0.0, places=6)
        self.assertAlmostEqual(a.noncontextual_fraction, 1.0, places=6)
        self.assertLessEqual(a.dephasing_robustness, a.atol)
        self.assertFalse(a.contextual)
        self.assertTrue(a.is_simplex_embeddable)

    def test_fraction_complement_and_bounds(self) -> None:
        a = NoncontextualityAssessment(_cabello_scenario(), monotone="contextual_fraction")
        self.assertAlmostEqual(a.contextual_fraction + a.noncontextual_fraction, 1.0, places=9)
        self.assertGreaterEqual(a.noncontextual_fraction, -1e-7)
        self.assertLessEqual(a.noncontextual_fraction, 1.0 + 1e-7)

    def test_clarabel_matches_mosek(self) -> None:
        # Small scenario: CLARABEL is reliable here (it can error on the large,
        # degenerate Cabello LP), so cross-solver parity is checked on this one.
        scenario = _noncontextual_222()
        mosek = NoncontextualityAssessment(scenario, monotone="both")
        clarabel = NoncontextualityAssessment(scenario, monotone="both", backend_solver="clarabel")
        self.assertAlmostEqual(mosek.contextual_fraction, clarabel.contextual_fraction, places=6)
        self.assertAlmostEqual(mosek.dephasing_robustness, clarabel.dephasing_robustness, places=6)


class WitnessInvariantTests(unittest.TestCase):
    def _check_witness(self, assessment: NoncontextualityAssessment) -> None:
        data = np.asarray(assessment.scenario.data_numeric, dtype=float)
        rays = _noncontextual_extremal_products(assessment)
        measures = {
            "contextual_fraction": assessment.contextual_fraction,
            "dephasing_robustness": assessment.dephasing_robustness,
        }
        for monotone, measure in measures.items():
            alpha = assessment.inequality[monotone]
            self.assertEqual(alpha.shape, data.shape)
            # Violation equals the optimizer's monotone value (strong duality).
            self.assertAlmostEqual(assessment.violation[monotone], measure, places=6)
            # Every noncontextual extremal ray respects the inequality's sense (bound 0).
            projections = rays @ alpha.reshape(-1)
            if assessment.inequality_sense[monotone] == "<=":
                self.assertTrue(np.all(projections <= assessment.inequality_bound[monotone] + 1e-7))
            else:
                self.assertTrue(np.all(projections >= assessment.inequality_bound[monotone] - 1e-7))

    def test_witness_on_cabello(self) -> None:
        self._check_witness(NoncontextualityAssessment(_cabello_scenario(), monotone="both"))

    def test_witness_on_noncontextual(self) -> None:
        self._check_witness(NoncontextualityAssessment(_noncontextual_222(), monotone="both"))

    def test_witness_on_intermediate_fraction(self) -> None:
        cabello = _cabello_scenario()
        P = np.asarray(cabello.data_numeric, dtype=float)
        mixed = 0.7 * P + 0.3 * _default_dephasing_target(P, atol=cabello.atol)
        assessment = NoncontextualityAssessment(ContextualityScenario(mixed), monotone="both")
        self.assertGreater(assessment.contextual_fraction, 1e-6)
        self.assertLess(assessment.contextual_fraction, 1.0 - 1e-6)
        self._check_witness(assessment)

    def test_dual_value_shapes(self) -> None:
        a = NoncontextualityAssessment(_cabello_scenario(), monotone="both")
        duals = a.dual_values
        self.assertEqual(duals["contextual_fraction"]["subbehavior_le_data"].shape, (a.num_x, a.num_y, a.num_b))
        self.assertEqual(duals["contextual_fraction"]["uniform_mass"].shape, (a.num_x, a.num_y))
        self.assertEqual(duals["dephasing_robustness"]["dephased_behavior"].shape, (a.num_x, a.num_y, a.num_b))


class ScenarioIntegrationTests(unittest.TestCase):
    def test_scenario_methods_match_assessment(self) -> None:
        scenario = _cabello_scenario()
        a = scenario.assess_noncontextuality()
        self.assertEqual(scenario.compute_contextual_fraction(), a.contextual_fraction)
        self.assertEqual(scenario.compute_noncontextual_fraction(), a.noncontextual_fraction)
        self.assertEqual(scenario.compute_dephasing_robustness(), a.dephasing_robustness)
        self.assertTrue(scenario.is_contextual())


if __name__ == "__main__":
    unittest.main()

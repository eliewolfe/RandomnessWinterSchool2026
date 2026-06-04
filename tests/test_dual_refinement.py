"""Tests for the L2-norm dual-certificate refinement.

These verify that the refined witness duals (a) reproduce strong duality
exactly, (b) lie at no-larger L2 norm than the raw LP-side duals, (c) preserve
sign constraints on the witness duals, and (d) thread their solver-selection
parameter through ContextualityScenario/ContextualityProtocol correctly.
"""

import unittest
import warnings

import numpy as np

from contextualityqkd.contextuality import NoncontextualityAssessment
from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.scenario import ContextualityScenario

from tests.test_contextuality import _cabello_scenario, _noncontextual_222


def _contextual_222() -> ContextualityScenario:
    """Small 2x2x2 scenario with nontrivial contextuality."""
    # PR-box-style violation, contextual under the GPT scenario abstraction.
    data = np.array(
        [
            [[1.0, 0.0], [0.5, 0.5]],
            [[0.5, 0.5], [0.0, 1.0]],
        ],
        dtype=float,
    )
    return ContextualityScenario(data)


class StrongDualityTests(unittest.TestCase):
    """After QP refinement, <alpha, data> must equal the primal measure."""

    def test_cabello_contextual_fraction(self) -> None:
        scenario = _cabello_scenario()
        data = np.asarray(scenario.data_numeric, dtype=float)
        assess = NoncontextualityAssessment(scenario, monotone="contextual_fraction")
        alpha = assess.inequality["contextual_fraction"]
        self.assertAlmostEqual(
            float(np.sum(alpha * data)),
            assess.contextual_fraction,
            places=6,
            msg="Refined contextual-fraction witness must satisfy <alpha, data> == measure.",
        )

    def test_cabello_dephasing_robustness(self) -> None:
        scenario = _cabello_scenario()
        data = np.asarray(scenario.data_numeric, dtype=float)
        assess = NoncontextualityAssessment(scenario, monotone="dephasing_robustness")
        alpha = assess.inequality["dephasing_robustness"]
        self.assertAlmostEqual(
            float(np.sum(alpha * data)),
            assess.dephasing_robustness,
            places=6,
        )

    def test_noncontextual_222_both_monotones(self) -> None:
        scenario = _noncontextual_222()
        data = np.asarray(scenario.data_numeric, dtype=float)
        assess = NoncontextualityAssessment(scenario, monotone="both")
        for monotone in ("contextual_fraction", "dephasing_robustness"):
            with self.subTest(monotone=monotone):
                alpha = assess.inequality[monotone]
                measure = (
                    assess.contextual_fraction
                    if monotone == "contextual_fraction"
                    else assess.dephasing_robustness
                )
                # Both monotones are zero on a noncontextual scenario; the
                # refined witness collapses to alpha == 0 and the identity
                # still holds trivially.
                self.assertAlmostEqual(float(np.sum(alpha * data)), float(measure), places=6)

    def test_eve_per_y_witness(self) -> None:
        scenario = _contextual_222()
        protocol = ContextualityProtocol(
            scenario, master_key_holder="Alice", lp_solver="mosek_simplex",
        )
        solver = protocol.eve_master_key_lp_solver
        data = np.asarray(scenario.data_numeric, dtype=float)
        guesses = solver.eve_guess_by_y
        for y, coeffs in solver.guess_bound_coeffs_by_y.items():
            with self.subTest(y=y):
                self.assertAlmostEqual(
                    float(np.sum(coeffs * data)), float(guesses[y]), places=6,
                )


class NormReductionTests(unittest.TestCase):
    """Refined witness duals must have L2 norm no larger than the raw duals."""

    def _norm_sq(self, duals: dict[str, np.ndarray]) -> float:
        return float(sum(np.sum(np.asarray(arr, dtype=float) ** 2) for arr in duals.values()))

    def test_cabello_contextual_fraction(self) -> None:
        scenario = _cabello_scenario()
        assess = NoncontextualityAssessment(scenario, monotone="contextual_fraction")
        raw = assess.raw_dual_values["contextual_fraction"]
        refined = assess.dual_values["contextual_fraction"]
        # Only the witness-relevant pair contributes to the objective; auxiliary
        # duals (e.g. the lambda<=1 multiplier) are not in either dict.
        raw_n = float(np.sum(raw["subbehavior_le_data"] ** 2) + np.sum(raw["uniform_mass"] ** 2))
        ref_n = float(
            np.sum(refined["subbehavior_le_data"] ** 2) + np.sum(refined["uniform_mass"] ** 2)
        )
        self.assertLessEqual(ref_n, raw_n + 1e-9)
        # The Cabello CF face is highly degenerate; verify the projection
        # actually moves (norm strictly decreases by a meaningful amount).
        self.assertLess(ref_n, raw_n * 0.5)

    def test_cabello_dephasing_robustness(self) -> None:
        scenario = _cabello_scenario()
        assess = NoncontextualityAssessment(scenario, monotone="dephasing_robustness")
        raw_n = float(np.sum(assess.raw_dual_values["dephasing_robustness"]["dephased_behavior"] ** 2))
        ref_n = float(np.sum(assess.dual_values["dephasing_robustness"]["dephased_behavior"] ** 2))
        self.assertLessEqual(ref_n, raw_n + 1e-9)
        self.assertLess(ref_n, raw_n * 0.5)

    def test_eve_per_y_norm_reduction(self) -> None:
        scenario = _contextual_222()
        protocol = ContextualityProtocol(
            scenario, master_key_holder="Alice", lp_solver="mosek_simplex",
        )
        solver = protocol.eve_master_key_lp_solver
        # Trigger refinement so both dicts are populated.
        refined = solver.guess_bound_coeffs_by_y
        raw = solver._raw_guess_bound_coeffs_by_y
        for y in refined:
            with self.subTest(y=y):
                self.assertLessEqual(
                    float(np.sum(refined[y] ** 2)),
                    float(np.sum(raw[y] ** 2)) + 1e-9,
                )


class WitnessSignTests(unittest.TestCase):
    """Refined witness duals must keep the LP-side sign constraints."""

    def test_contextual_fraction_mu_nonneg(self) -> None:
        scenario = _cabello_scenario()
        assess = NoncontextualityAssessment(scenario, monotone="contextual_fraction")
        mu = assess.dual_values["contextual_fraction"]["subbehavior_le_data"]
        self.assertGreaterEqual(float(mu.min()), -1e-7)

    def test_eve_c_y_nonneg(self) -> None:
        scenario = _contextual_222()
        protocol = ContextualityProtocol(
            scenario, master_key_holder="Alice", lp_solver="mosek_simplex",
        )
        for y, coeffs in protocol.eve_master_key_lp_solver.guess_bound_coeffs_by_y.items():
            with self.subTest(y=y):
                self.assertGreaterEqual(float(coeffs.min()), -1e-7)


class QpSolverFallbackTests(unittest.TestCase):
    """LP-only and unknown solver tokens must degrade gracefully."""

    def test_lp_only_token_falls_through_with_warning(self) -> None:
        scenario = _cabello_scenario()
        # qp_solver='highs' is LP-only; resolver must warn and fall through
        # to the default Gurobi/Mosek/Clarabel/OSQP/SCS chain.
        assess = NoncontextualityAssessment(scenario, monotone="contextual_fraction", qp_solver="highs")
        with warnings.catch_warnings(record=True) as warning_log:
            warnings.simplefilter("always")
            alpha = assess.inequality["contextual_fraction"]
        messages = [str(w.message) for w in warning_log]
        self.assertTrue(
            any("qp_solver='highs'" in m and "LP-only" in m for m in messages),
            msg=f"Expected an LP-only warning for qp_solver='highs', got: {messages}",
        )
        data = np.asarray(scenario.data_numeric, dtype=float)
        self.assertAlmostEqual(float(np.sum(alpha * data)), assess.contextual_fraction, places=6)


class QpSolverParameterThreadingTests(unittest.TestCase):
    """The qp_solver parameter must reach every nested entity that consumes it."""

    def test_protocol_to_lp_solver(self) -> None:
        scenario = _noncontextual_222()
        protocol = ContextualityProtocol(
            scenario, master_key_holder="Alice", lp_solver="mosek_simplex", qp_solver="clarabel",
        )
        self.assertEqual(protocol.qp_solver, "clarabel")
        self.assertEqual(protocol.eve_master_key_lp_solver.qp_solver, "clarabel")

    def test_scenario_to_assessment(self) -> None:
        scenario = _noncontextual_222()
        assess = scenario.assess_noncontextuality(qp_solver="osqp")
        self.assertEqual(assess.qp_solver, "osqp")


class RawDualsPreservedTests(unittest.TestCase):
    """raw_dual_values / _raw_guess_bound_coeffs_by_y survive refinement."""

    def test_assessment_raw_values_match_lp_snapshot(self) -> None:
        scenario = _cabello_scenario()
        assess = NoncontextualityAssessment(scenario, monotone="contextual_fraction")
        # Touch both surfaces; raw must remain raw even after refinement runs.
        _ = assess.dual_values
        raw_mu = assess.raw_dual_values["contextual_fraction"]["subbehavior_le_data"]
        ref_mu = assess.dual_values["contextual_fraction"]["subbehavior_le_data"]
        # On Cabello the dual face is degenerate, so the two arrays differ.
        self.assertFalse(np.allclose(raw_mu, ref_mu))

    def test_eve_solver_raw_attribute_populated(self) -> None:
        scenario = _contextual_222()
        protocol = ContextualityProtocol(
            scenario, master_key_holder="Alice", lp_solver="mosek_simplex",
        )
        solver = protocol.eve_master_key_lp_solver
        # solve_lp() runs eagerly via the cached_property; the raw snapshot must
        # already be filled before we touch the refined public accessor.
        self.assertTrue(len(solver._raw_guess_bound_coeffs_by_y) >= 1)
        self.assertTrue(len(solver._raw_dual_values_by_y) >= 1)


if __name__ == "__main__":
    unittest.main()

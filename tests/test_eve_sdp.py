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
        self.assertEqual(solver.template.real_dimension, 2 * solver.template.dimension)

    def test_complex_real_embedding_roundtrip(self) -> None:
        complex_matrix = np.array(
            [
                [1.0 + 0.0j, 0.25 + 0.5j],
                [0.25 - 0.5j, 2.0 + 0.0j],
            ],
            dtype=complex,
        )
        embedded = QKDNoncontextualSDP._complex_to_real_block(complex_matrix)
        recovered = QKDNoncontextualSDP._real_block_to_complex(embedded)

        self.assertEqual(embedded.shape, (4, 4))
        np.testing.assert_allclose(recovered, complex_matrix)

    def test_nonprojective_template_is_larger_by_operator_doubling_and_realification(self) -> None:
        data = np.array([[[0.5, 0.3, 0.2]]], dtype=float)
        scenario = ContextualityScenario(data)

        projective = QKDNoncontextualSDP(
            scenario,
            projective_bob=True,
            projective_eve=True,
            npa_level_bob=1,
            npa_level_eve=1,
        )
        nonprojective = QKDNoncontextualSDP(
            scenario,
            projective_bob=False,
            projective_eve=False,
            npa_level_bob=1,
            npa_level_eve=1,
        )

        projective.instantiate_moment_matrices()
        nonprojective.instantiate_moment_matrices()

        self.assertIsNotNone(projective.template)
        self.assertIsNotNone(nonprojective.template)

        # Y=1, K=3, level-(1,1):
        # projective primitive letters = K-1 = 2 per party, so d = 1 + 2 + 2 + 2*2 = 9
        # nonprojective primitive letters = 2*(K-1)=4 per party, so d = 1 + 4 + 4 + 4*4 = 25
        self.assertEqual(projective.template.dimension, 9)
        self.assertEqual(nonprojective.template.dimension, 25)
        self.assertEqual(projective.template.real_dimension, 18)
        self.assertEqual(nonprojective.template.real_dimension, 50)
        self.assertGreater(nonprojective.template.dimension, projective.template.dimension)
        self.assertEqual(nonprojective.template.real_dimension, 2 * nonprojective.template.dimension)

    def test_projective_orthogonal_projector_products_are_zero_including_reconstructed_last_outcome(self) -> None:
        data = np.array([[[0.5, 0.3, 0.2]]], dtype=float)
        scenario = ContextualityScenario(data)
        solver = QKDNoncontextualSDP(
            scenario,
            projective_bob=True,
            projective_eve=True,
            npa_level_bob=1,
            npa_level_eve=1,
        )
        solver.instantiate_moment_matrices()

        self.assertIsNotNone(solver.template)
        template = solver.template

        p0 = template.find_operator("B", 0, 0)
        p1 = template.find_operator("B", 0, 1)
        self.assertIsNotNone(p0)
        self.assertIsNotNone(p1)

        # Primitive orthogonal projectors of the same setting annihilate.
        self.assertIsNone(template.word_index((p0, p1)))
        self.assertIsNone(template.word_index((p1, p0)))

        # The omitted final outcome is reconstructed as I - P0 - P1, so it is
        # also orthogonal to each kept primitive projector.
        for kept in (p0, p1):
            coeffs_by_col: dict[int, float] = {}
            for word, coeff in solver._effect_affine_words("B", 0, 2):
                col = template.word_index(word + (kept,))
                if col is None:
                    continue
                coeffs_by_col[col] = coeffs_by_col.get(col, 0.0) + float(coeff)
            surviving = {col: value for col, value in coeffs_by_col.items() if abs(value) > 1e-9}
            self.assertEqual(surviving, {}, f"Reconstructed last outcome should be orthogonal to kept projector {kept}")

    def test_solves_toy_eve_success_bound(self) -> None:
        data = np.array([[[0.8, 0.2]]], dtype=float)
        scenario = ContextualityScenario(data)
        solver = QKDNoncontextualSDP(scenario, npa_level_bob=1, npa_level_eve=1, threads=1)

        value = solver.solve_sdp()

        self.assertTrue(np.isfinite(value))
        self.assertGreaterEqual(value, -1e-7)
        self.assertLessEqual(value, 1.0 + 1e-6)
        self.assertTrue(np.isfinite(solver.key_rate_lower_bound))
        self.assertEqual(len(solver.solution_matrices_real), scenario.X_cardinality)
        self.assertEqual(len(solver.solution_matrices), scenario.X_cardinality)
        self.assertEqual(solver.solution_matrices[0].shape[0] * 2, solver.solution_matrices_real[0].shape[0])

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

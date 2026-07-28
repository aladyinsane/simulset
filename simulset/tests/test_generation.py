import random
import unittest

import numpy as np

from ..generation import categorical, copula, identifiers, missingness, numeric


class TestNumeric(unittest.TestCase):
    def test_empirical_cdf_sample_bounds(self):
        grid = np.linspace(0.0, 100.0, 1001)
        u = np.array([0.0, 0.5, 1.0])
        result = numeric.empirical_cdf_sample(grid, u)
        self.assertAlmostEqual(result[0], 0.0, places=6)
        self.assertAlmostEqual(result[1], 50.0, places=1)
        self.assertAlmostEqual(result[2], 100.0, places=6)

    def test_format_numeric_integer(self):
        values = np.array([1.4, 2.6, 3.5])
        result = numeric.format_numeric(values, "integer", 0)
        self.assertTrue(np.array_equal(result, np.array([1, 3, 4])))

    def test_format_numeric_decimals(self):
        values = np.array([1.005, 2.999])
        result = numeric.format_numeric(values, "float", 2)
        self.assertTrue(np.allclose(result, np.array([1.0, 3.0]), atol=0.02))


class TestCategorical(unittest.TestCase):
    def test_generate_never_emits_real_labels(self):
        col_profile = {"frequencies": {"orange": 0.5, "red": 0.3, "green": 0.2}}
        rng = np.random.default_rng(42)
        chosen, mapping = categorical.generate("color", col_profile, 1000, rng)
        real_labels = set(col_profile["frequencies"].keys())
        self.assertTrue(all(v not in real_labels for v in chosen))
        self.assertEqual(set(mapping.keys()), real_labels)
        self.assertEqual(set(chosen.tolist()), set(mapping.values()))

    def test_proportions_roughly_preserved(self):
        col_profile = {"frequencies": {"a": 0.8, "b": 0.2}}
        rng = np.random.default_rng(1)
        chosen, mapping = categorical.generate("x", col_profile, 20000, rng)
        share_a = sum(1 for v in chosen if v == mapping["a"]) / len(chosen)
        self.assertAlmostEqual(share_a, 0.8, delta=0.02)


class TestIdentifiers(unittest.TestCase):
    def test_pattern_generated_values_match_regex(self):
        import re
        col_profile = {
            "pattern_template": [("literal", "P"), ("class", "digit", 6)],
            "length_stats": {"min": 7, "max": 7},
        }
        rng = random.Random(0)
        values = identifiers.generate("id", col_profile, 100, rng, enforce_unique=True)
        self.assertEqual(len(values), 100)
        self.assertEqual(len(set(values)), 100)
        for v in values:
            self.assertTrue(re.match(r"^P[0-9]{6}$", v))

    def test_fallback_when_no_template(self):
        col_profile = {"pattern_template": None, "length_stats": {"min": 5, "max": 5}}
        rng = random.Random(0)
        values = identifiers.generate("x", col_profile, 20, rng, enforce_unique=False)
        self.assertTrue(all(len(v) == 5 for v in values))


class TestMissingness(unittest.TestCase):
    def test_injects_correct_count(self):
        rng = random.Random(0)
        values = list(range(1000))
        result = missingness.inject(values, 0.1, rng)
        n_missing = sum(1 for v in result if v is None)
        self.assertEqual(n_missing, 100)

    def test_zero_rate_leaves_values_untouched(self):
        rng = random.Random(0)
        values = list(range(50))
        result = missingness.inject(values, 0.0, rng)
        self.assertEqual(result, values)


class TestCopula(unittest.TestCase):
    def test_identity_matrix_gives_independent_uniforms(self):
        rng = np.random.default_rng(0)
        corr = np.eye(3)
        u = copula.draw_correlated_uniforms(5000, corr, rng)
        self.assertEqual(u.shape, (5000, 3))
        self.assertTrue(np.all(u >= 0) and np.all(u <= 1))

    def test_high_correlation_columns_track_each_other(self):
        rng = np.random.default_rng(0)
        corr = np.array([[1.0, 0.9], [0.9, 1.0]])
        u = copula.draw_correlated_uniforms(20000, corr, rng)
        observed_corr = np.corrcoef(u[:, 0], u[:, 1])[0, 1]
        self.assertGreater(observed_corr, 0.7)


if __name__ == "__main__":
    unittest.main()

import random
import unittest

from ..profiling import classify, patterns, stats


class TestClassify(unittest.TestCase):
    def test_numeric_column(self):
        cls = classify.classify_column("age", ["1", "2", "3", "", "4", "1", "2", "3", "4", "2"])
        self.assertEqual(cls.dtype, classify.NUMERIC)
        self.assertEqual(cls.subtype, "integer")
        self.assertEqual(cls.n_non_null, 9)

    def test_float_column(self):
        cls = classify.classify_column("cost", ["1.50", "2.00", "3.75"])
        self.assertEqual(cls.dtype, classify.NUMERIC)
        self.assertEqual(cls.subtype, "float")

    def test_date_column(self):
        cls = classify.classify_column("d", ["2020-01-01", "2020-06-15", "2021-12-31"])
        self.assertEqual(cls.dtype, classify.DATE)

    def test_categorical_column(self):
        cls = classify.classify_column("color", ["red", "green", "red", "blue", "red"])
        self.assertEqual(cls.dtype, classify.CATEGORICAL)

    def test_identifier_column(self):
        cls = classify.classify_column("id", [f"P{i:06d}" for i in range(200)])
        self.assertEqual(cls.dtype, classify.IDENTIFIER)
        self.assertTrue(cls.is_fully_unique)

    def test_free_text_column(self):
        names = [f"name-{i}" for i in range(200)] + [f"name-{i}" for i in range(50)]
        cls = classify.classify_column("name", names)
        self.assertEqual(cls.dtype, classify.FREE_TEXT)

    def test_sas_missing_marker(self):
        cls = classify.classify_column("x", ["1", ".", "2", "3", "1", "2", "3", "1", "2", "3", "1", "2", "3", "1", "2", "3", "1", "2", "3", "1", "2"])
        self.assertEqual(cls.dtype, classify.NUMERIC)
        self.assertEqual(cls.n_non_null, 20)

    def test_all_missing_defaults_numeric(self):
        cls = classify.classify_column("x", ["", "", "."])
        self.assertEqual(cls.dtype, classify.NUMERIC)
        self.assertEqual(cls.n_non_null, 0)


class TestPatterns(unittest.TestCase):
    def test_uniform_pattern(self):
        rng = random.Random(0)
        values = [f"P{rng.randint(0, 999999):06d}" for _ in range(200)]
        result = patterns.infer_pattern(values)
        self.assertTrue(result["uniform_length"])
        self.assertEqual(result["regex"], "^P[0-9]{6}$")

    def test_non_uniform_length_has_no_regex(self):
        result = patterns.infer_pattern(["ab", "abc", "abcd"])
        self.assertFalse(result["uniform_length"])
        self.assertIsNone(result["regex"])


class TestStats(unittest.TestCase):
    def test_numeric_stats_basic(self):
        raw = ["1", "2", "3", "4", "5"]
        parsed = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = stats.numeric_stats("x", parsed, raw)
        self.assertEqual(result["min"], 1.0)
        self.assertEqual(result["max"], 5.0)
        self.assertEqual(result["mean"], 3.0)
        self.assertEqual(result["median"], 3.0)
        self.assertEqual(result["decimals"], 0)
        self.assertEqual(len(result["empirical_quantiles"]), stats.N_EMPIRICAL_QUANTILES)

    def test_decimals_inferred_from_raw_strings(self):
        raw = ["19.99", "20.00", "5.5"]
        parsed = [19.99, 20.00, 5.5]
        result = stats.numeric_stats("x", parsed, raw)
        self.assertEqual(result["decimals"], 2)

    def test_categorical_frequencies_sum_to_one(self):
        result = stats.categorical_stats(["red", "red", "blue", "green", "red"])
        self.assertAlmostEqual(sum(result["frequencies"].values()), 1.0)
        self.assertEqual(result["frequencies"]["red"], 3 / 5)

    def test_correlation_matrix_perfect_correlation(self):
        x = list(range(1, 101))
        y = [v * 2.0 for v in x]
        matrix = stats.compute_correlation_matrix({"x": [float(v) for v in x], "y": y})
        self.assertIsNotNone(matrix)
        self.assertAlmostEqual(matrix["matrix"][0][1], 1.0, places=6)

    def test_correlation_matrix_skips_constant_column(self):
        x = [float(v) for v in range(1, 51)]
        constant = [1.0] * 50
        matrix = stats.compute_correlation_matrix({"x": x, "const": constant})
        self.assertIsNone(matrix)


if __name__ == "__main__":
    unittest.main()

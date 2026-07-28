import math
import unittest

import numpy as np

from ..mathutils import erf_np, inv_norm_cdf_np, norm_cdf_np, normal_scores, rankdata_avg


class TestErf(unittest.TestCase):
    def test_matches_math_erf(self):
        xs = np.linspace(-4, 4, 41)
        approx = erf_np(xs)
        exact = np.array([math.erf(x) for x in xs])
        self.assertTrue(np.allclose(approx, exact, atol=1e-6))


class TestNormCdf(unittest.TestCase):
    def test_zero_is_half(self):
        self.assertAlmostEqual(float(norm_cdf_np(np.array([0.0]))[0]), 0.5, places=6)

    def test_monotonic(self):
        xs = np.linspace(-3, 3, 25)
        ys = norm_cdf_np(xs)
        self.assertTrue(np.all(np.diff(ys) > 0))


class TestInvNormCdf(unittest.TestCase):
    def test_round_trip_with_norm_cdf(self):
        ps = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
        z = inv_norm_cdf_np(ps)
        back = norm_cdf_np(z)
        self.assertTrue(np.allclose(back, ps, atol=1e-6))

    def test_median_is_zero(self):
        self.assertAlmostEqual(float(inv_norm_cdf_np(np.array([0.5]))[0]), 0.0, places=6)


class TestRankdata(unittest.TestCase):
    def test_no_ties(self):
        values = np.array([30.0, 10.0, 20.0])
        ranks = rankdata_avg(values)
        self.assertTrue(np.array_equal(ranks, np.array([3.0, 1.0, 2.0])))

    def test_ties_get_average_rank(self):
        values = np.array([10.0, 10.0, 20.0])
        ranks = rankdata_avg(values)
        self.assertTrue(np.array_equal(ranks, np.array([1.5, 1.5, 3.0])))


class TestNormalScores(unittest.TestCase):
    def test_preserves_rank_order(self):
        values = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        scores = normal_scores(values)
        self.assertTrue(np.array_equal(np.argsort(values), np.argsort(scores)))


if __name__ == "__main__":
    unittest.main()

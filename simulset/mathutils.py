"""Vectorized normal-distribution and rank helpers shared by profiling (correlation
matrix / normal scores) and generation (Gaussian copula). No scipy dependency -
erf and the inverse normal CDF are both rational/polynomial approximations,
fully vectorized with numpy so they scale to hundreds of thousands of rows.
"""

from __future__ import annotations

import numpy as np


def erf_np(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun 7.1.26 approximation of erf. Max abs error ~1.5e-7."""
    a1, a2, a3, a4, a5 = (
        0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429,
    )
    p = 0.3275911
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-ax * ax)
    return sign * y


def norm_cdf_np(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF, vectorized."""
    return 0.5 * (1.0 + erf_np(z / np.sqrt(2.0)))


def inv_norm_cdf_np(p: np.ndarray) -> np.ndarray:
    """Inverse standard normal CDF (probit), Acklam's rational approximation.
    Accurate to ~1.15e-9 relative error, fully vectorized (no scipy)."""
    a = [
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
        3.754408661907416e00,
    ]
    p_low = 0.02425
    p_high = 1 - p_low

    p = np.clip(np.asarray(p, dtype=float), 1e-10, 1 - 1e-10)

    q_low = np.sqrt(-2 * np.log(np.where(p < p_low, p, p_low)))
    x_low = (
        (((((c[0] * q_low + c[1]) * q_low + c[2]) * q_low + c[3]) * q_low + c[4]) * q_low + c[5])
        / ((((d[0] * q_low + d[1]) * q_low + d[2]) * q_low + d[3]) * q_low + 1)
    )

    q_mid = p - 0.5
    r = q_mid * q_mid
    x_mid = (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q_mid
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )

    q_high = np.sqrt(-2 * np.log(np.where(p > p_high, 1 - p, 1 - p_high)))
    x_high = -(
        (((((c[0] * q_high + c[1]) * q_high + c[2]) * q_high + c[3]) * q_high + c[4]) * q_high + c[5])
        / ((((d[0] * q_high + d[1]) * q_high + d[2]) * q_high + d[3]) * q_high + 1)
    )

    return np.where(p < p_low, x_low, np.where(p > p_high, x_high, x_mid))


def rankdata_avg(values: np.ndarray) -> np.ndarray:
    """Average rank (1-indexed, ties get the mean rank of their group)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    sorter = np.argsort(values, kind="mergesort")
    sorted_vals = values[sorter]

    is_new_group = np.r_[True, sorted_vals[1:] != sorted_vals[:-1]]
    group_id = np.cumsum(is_new_group) - 1

    positions = np.arange(1, n + 1)
    n_groups = group_id[-1] + 1
    sum_ranks = np.zeros(n_groups)
    counts = np.zeros(n_groups)
    np.add.at(sum_ranks, group_id, positions)
    np.add.at(counts, group_id, 1)
    avg_rank_per_group = sum_ranks / counts

    avg_rank_sorted = avg_rank_per_group[group_id]
    avg_rank = np.empty(n)
    avg_rank[sorter] = avg_rank_sorted
    return avg_rank


def normal_scores(values: np.ndarray) -> np.ndarray:
    """Rank-transform values to standard-normal scores (Sec 6.3 step 1)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    ranks = rankdata_avg(values)
    p = (ranks - 0.5) / n
    return inv_norm_cdf_np(p)

"""Empirical-CDF interpolation sampler for numeric (and date-as-epoch-day)
columns (Sec 6.2)."""

from __future__ import annotations

import numpy as np


def empirical_cdf_sample(quantile_grid: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse-transform sampling against a pre-computed empirical quantile
    grid: for each u in [0, 1], linearly interpolate between the two grid
    points bracketing u's rank. quantile_grid is assumed to be evenly spaced
    over [0, 1] (as produced by profiling.stats numeric_stats/date_stats)."""
    grid = np.asarray(quantile_grid, dtype=float)
    n = len(grid)
    if n == 1:
        return np.full_like(np.asarray(u, dtype=float), grid[0])
    xp = np.linspace(0.0, 1.0, n)
    return np.interp(u, xp, grid)


def format_numeric(values: np.ndarray, subtype: str, decimals: int) -> np.ndarray:
    """Round/cast generated values back to the profiled precision (Sec 6.5)."""
    if subtype == "integer":
        return np.round(values).astype(np.int64)
    return np.round(values, decimals)

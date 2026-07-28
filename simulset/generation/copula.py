"""Gaussian copula: draw correlated uniforms for the non-flagged numeric
columns so their empirical-CDF sampling (numeric.py) preserves the real
cross-column correlation structure (Sec 6.3)."""

from __future__ import annotations

import logging

import numpy as np

from ..mathutils import norm_cdf_np

log = logging.getLogger(__name__)


def draw_correlated_uniforms(
    n_records: int, corr_matrix: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Returns an (n_records, k) array of Uniform(0,1) draws whose underlying
    normal scores share corr_matrix's correlation structure. Column order
    matches corr_matrix's row/column order."""
    k = corr_matrix.shape[0]
    try:
        chol = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        log.warning(
            "Correlation matrix is not positive-definite; adding a small "
            "jitter to the diagonal before Cholesky decomposition"
        )
        jitter = np.eye(k) * 1e-6
        chol = np.linalg.cholesky(corr_matrix + jitter)

    z_independent = rng.standard_normal(size=(n_records, k))
    z_correlated = z_independent @ chol.T
    return norm_cdf_np(z_correlated)

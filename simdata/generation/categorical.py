"""Synthetic-label categorical sampler (Sec 6.2 non-flagged categorical strategy)."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

log = logging.getLogger(__name__)


def build_synthetic_labels(col_name: str, real_labels: List[str]) -> List[str]:
    """One synthetic label per distinct real value, ordered by descending
    real frequency (real_labels is assumed already sorted that way, since
    categorical_stats builds its frequency dict via Counter.most_common)."""
    return [f"{col_name}_val{i + 1}" for i in range(len(real_labels))]


def generate(
    col_name: str,
    col_profile: Dict,
    n_records: int,
    rng: np.random.Generator,
    randomize_labels: bool = False,
) -> Tuple[np.ndarray, Dict[str, str]]:
    """Returns (generated synthetic labels, real->synthetic mapping)."""
    frequencies = col_profile["frequencies"]
    real_labels = list(frequencies.keys())
    weights = np.array(list(frequencies.values()), dtype=float)
    weights = weights / weights.sum()

    synthetic_labels = build_synthetic_labels(col_name, real_labels)
    if randomize_labels:
        perm = rng.permutation(len(synthetic_labels))
        synthetic_labels = [synthetic_labels[i] for i in perm]

    chosen = rng.choice(np.array(synthetic_labels, dtype=object), size=n_records, p=weights)
    mapping = dict(zip(real_labels, synthetic_labels))
    return chosen, mapping

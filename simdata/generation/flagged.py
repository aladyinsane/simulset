"""Bootstrap sampling for --use-real-values columns (Sec 6.2, "Flagged columns").

Sample directly from real observed values in their real proportions. For
categorical columns the frequency table already embedded in profile.json is
enough. For every other dtype, generate_dataset.py must re-read the raw
values from --real-csv (profile.json doesn't retain full raw value lists for
non-categorical columns - see the module docstring in generate_dataset.py).
"""

from __future__ import annotations

import random
from typing import Dict, List

import numpy as np


def generate_categorical_flagged(col_profile: Dict, n_records: int, rng: np.random.Generator) -> np.ndarray:
    frequencies = col_profile["frequencies"]
    labels = list(frequencies.keys())
    weights = np.array(list(frequencies.values()), dtype=float)
    weights = weights / weights.sum()
    return rng.choice(np.array(labels, dtype=object), size=n_records, p=weights)


def generate_from_raw_values(raw_values: List[str], n_records: int, rng: random.Random) -> List[str]:
    """Bootstrap-sample literal values (numeric/date/identifier/free_text
    flagged columns) uniformly from the observed raw values - equivalent to
    random.choices with equal weights, which reproduces the real proportions
    automatically since each real occurrence is present in raw_values once."""
    return rng.choices(raw_values, k=n_records)

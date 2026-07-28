"""Per-column missing-value injection (Sec 6.4)."""

from __future__ import annotations

import random
from typing import List, Optional, Sequence


def inject(values: Sequence, missing_rate: float, rng: random.Random) -> List[Optional[object]]:
    """Randomly null out round(missing_rate * n) of the given values.
    Missingness is treated as independent per column (joint/co-missingness
    is a Phase 2 stretch goal per Sec 10)."""
    n = len(values)
    result = list(values)
    if missing_rate <= 0 or n == 0:
        return result

    n_missing = round(missing_rate * n)
    n_missing = min(n_missing, n)
    if n_missing == 0:
        return result

    missing_indices = rng.sample(range(n), n_missing)
    for i in missing_indices:
        result[i] = None
    return result

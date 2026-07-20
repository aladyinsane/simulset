"""Date column generation: empirical-CDF interpolation on the epoch-day
representation (Sec 6.2, "DATE columns"). Weekday/month re-weighting is a
Phase 2 stretch goal (Sec 10) and is not implemented here."""

from __future__ import annotations

from datetime import date
from typing import Dict, List

import numpy as np

from . import numeric


def generate(col_profile: Dict, u: np.ndarray) -> List[str]:
    quantile_grid = col_profile["empirical_quantiles_epoch_day"]
    ordinals = numeric.empirical_cdf_sample(quantile_grid, u)
    ordinals = np.round(ordinals).astype(np.int64)
    return [date.fromordinal(int(o)).isoformat() for o in ordinals]

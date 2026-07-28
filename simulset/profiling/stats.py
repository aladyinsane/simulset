"""Per-column statistics (Sec 4, Sec 5.2) and the cross-column correlation matrix
(Sec 6.3 step 1-2)."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..mathutils import normal_scores
from . import patterns

log = logging.getLogger(__name__)

# Number of equally-spaced quantile points stored per numeric/date column so
# generate_dataset.py can do high-fidelity empirical-CDF sampling (Sec 6.2)
# without re-reading the real CSV. The plan's example schema (Sec 5.2) only
# shows 6 headline percentiles for human inspection; those are kept too, but
# 6 points is far too coarse a grid to sample from, so this profiler stores a
# denser one. 1001 floats/column is small (~a few hundred KB of JSON even at
# 100 numeric columns) and keeps profile.json fully self-contained, matching
# the "no need to re-read the real dataset" goal in Sec 3.
N_EMPIRICAL_QUANTILES = 1001

PERCENTILE_POINTS = (1, 5, 25, 75, 95, 99)


def _decimals_from_raw(raw_values: List[str]) -> int:
    max_dp = 0
    for v in raw_values:
        v = v.strip()
        if "." in v:
            dp = len(v.split(".", 1)[1].rstrip())
            max_dp = max(max_dp, dp)
    return min(max_dp, 10)


def numeric_stats(name: str, parsed_values: List[float], raw_values: List[str]) -> Dict:
    arr = np.asarray(parsed_values, dtype=float)
    n = len(arr)

    mean = float(np.mean(arr))
    variance = float(np.var(arr, ddof=1)) if n > 1 else 0.0
    m2 = float(np.mean((arr - mean) ** 2))
    m3 = float(np.mean((arr - mean) ** 3))
    m4 = float(np.mean((arr - mean) ** 4))
    skewness = m3 / (m2 ** 1.5) if m2 > 0 else 0.0
    kurtosis = m4 / (m2 ** 2) - 3.0 if m2 > 0 else 0.0

    percentiles = {f"p{p}": float(np.percentile(arr, p)) for p in PERCENTILE_POINTS}
    quantile_grid = np.quantile(arr, np.linspace(0.0, 1.0, N_EMPIRICAL_QUANTILES)).tolist()

    return {
        "dtype": "numeric",
        "n_non_null": n,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": mean,
        "median": float(np.median(arr)),
        "variance": variance,
        "percentiles": percentiles,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "decimals": _decimals_from_raw(raw_values),
        "empirical_quantiles": quantile_grid,
    }


def categorical_stats(values: List[str]) -> Dict:
    counts = Counter(values)
    n = len(values)
    ordered = counts.most_common()
    return {
        "dtype": "categorical",
        "n_non_null": n,
        "n_unique": len(counts),
        "frequencies": {label: count / n for label, count in ordered},
    }


WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def date_stats(values: List[datetime]) -> Dict:
    n = len(values)
    ordinals = np.array([v.toordinal() for v in values], dtype=float)
    weekday_counts = Counter(v.weekday() for v in values)
    weekday_distribution = {
        WEEKDAY_NAMES[i]: weekday_counts.get(i, 0) / n for i in range(7)
    }
    quantile_grid = np.quantile(ordinals, np.linspace(0.0, 1.0, N_EMPIRICAL_QUANTILES)).tolist()

    min_dt, max_dt = min(values), max(values)
    return {
        "dtype": "date",
        "n_non_null": n,
        "min": min_dt.date().isoformat(),
        "max": max_dt.date().isoformat(),
        "weekday_distribution": weekday_distribution,
        "empirical_quantiles_epoch_day": quantile_grid,
    }


def identifier_stats(name: str, values: List[str], n_unique: int, is_fully_unique: bool) -> Dict:
    pattern = patterns.infer_pattern(values)
    lengths = [len(v) for v in values]
    if not pattern["uniform_length"]:
        log.warning(
            "%s: identifier values are not a uniform length (%d-%d chars); "
            "falling back to length-bounded random generation instead of a "
            "positional pattern",
            name, pattern["length_min"], pattern["length_max"],
        )
    return {
        "dtype": "identifier",
        "n_non_null": len(values),
        "n_unique": n_unique,
        "is_fully_unique": is_fully_unique,
        "value_pattern": pattern["regex"],
        "pattern_template": pattern["template"],
        "length_stats": {"min": min(lengths), "max": max(lengths)},
    }


def free_text_stats(name: str, values: List[str], n_unique: int) -> Dict:
    pattern = patterns.infer_pattern(values)
    lengths = [len(v) for v in values]
    return {
        "dtype": "free_text",
        "n_non_null": len(values),
        "n_unique": n_unique,
        "value_pattern": pattern["regex"],
        "pattern_template": pattern["template"],
        "length_stats": {"min": min(lengths), "max": max(lengths)},
    }


def compute_correlation_matrix(
    numeric_columns: Dict[str, List[Optional[float]]],
) -> Optional[Dict]:
    """Cross-column correlation matrix of standard-normal rank scores (Sec 6.3
    step 1-2). Restricted to complete-case rows (every included numeric column
    non-missing) and to non-constant columns, since a zero-variance column has
    no well-defined rank correlation. Returns None (and logs why) if fewer
    than two columns qualify.
    """
    candidate_names = list(numeric_columns.keys())
    if len(candidate_names) < 2:
        log.info("Correlation matrix skipped: fewer than 2 numeric columns present")
        return None

    n_rows = len(next(iter(numeric_columns.values())))
    complete_mask = np.ones(n_rows, dtype=bool)
    for col in candidate_names:
        col_mask = np.array([v is not None for v in numeric_columns[col]])
        complete_mask &= col_mask

    n_complete = int(complete_mask.sum())
    if n_complete < 2:
        log.warning(
            "Correlation matrix skipped: fewer than 2 rows with all numeric "
            "columns present simultaneously (%d complete rows)", n_complete,
        )
        return None

    usable_names = []
    score_columns = []
    for col in candidate_names:
        arr = np.array(
            [v for v, keep in zip(numeric_columns[col], complete_mask) if keep],
            dtype=float,
        )
        if np.std(arr) == 0:
            log.warning("%s: constant numeric column, excluded from correlation matrix", col)
            continue
        usable_names.append(col)
        score_columns.append(normal_scores(arr))

    if len(usable_names) < 2:
        log.warning("Correlation matrix skipped: fewer than 2 non-constant numeric columns")
        return None

    score_matrix = np.column_stack(score_columns)
    corr = np.corrcoef(score_matrix, rowvar=False)

    log.info(
        "Correlation matrix computed over %d numeric columns using %d complete-case rows",
        len(usable_names), n_complete,
    )
    return {"columns": usable_names, "matrix": corr.tolist()}

"""Column type classification (Sec 5.1 of the plan)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

log = logging.getLogger(__name__)

NUMERIC = "numeric"
DATE = "date"
CATEGORICAL = "categorical"
IDENTIFIER = "identifier"
FREE_TEXT = "free_text"

# Sentinels commonly used for "missing" in CSVs exported from SAS (which uses "."
# for a missing numeric value) and other common conventions.
MISSING_MARKERS = {"", "na", "n/a", "null", "none", "."}

DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%y",
    "%Y%m%d",
    "%d-%b-%Y",
]

# Fraction of non-null values that must parse cleanly for a column to be
# classified as numeric/date; a handful of stray bad values are tolerated
# and logged rather than demoting the whole column to a string type.
PARSE_THRESHOLD = 0.98


def is_missing(raw: Optional[str]) -> bool:
    if raw is None:
        return True
    return raw.strip().lower() in MISSING_MARKERS


def try_parse_number(raw: str) -> Optional[float]:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def looks_like_int(raw: str) -> bool:
    parsed = try_parse_number(raw)
    if parsed is None or parsed != int(parsed):
        return False
    # "20.0" parses as an int value but visually carries decimals; only
    # strings with no fractional part in their own text count as integer-formatted.
    return "." not in raw or raw.strip().split(".")[1].strip("0") == ""


def try_parse_date(raw: str) -> Optional[datetime]:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw.strip(), fmt)
        except ValueError:
            continue
    return None


@dataclass
class ColumnClassification:
    dtype: str
    subtype: Optional[str] = None
    n_non_null: int = 0
    n_unique: int = 0
    is_fully_unique: bool = False
    cardinality_ratio: float = 0.0
    non_null_values: List[str] = field(default_factory=list)


# distinct / non-missing at or above this makes an all-digit column a
# pattern-generated identifier rather than a categorical code.
NEAR_UNIQUE_RATIO = 0.95
# Same-length all-digit columns this long or longer are codes, not numbers.
MIN_FIXED_CODE_LENGTH = 5


def integer_code_reason(values: List[str]) -> Optional[str]:
    """If `values` (non-missing, trimmed) is an integer-formatted ID/code column,
    return why; otherwise None. Decimals, negatives and short variable-length
    numbers return None and stay numeric."""
    if not values or not all(v.isascii() and v.isdigit() for v in values):
        return None
    if any(len(v) > 1 and v.startswith("0") for v in values):
        return "has leading zeros"
    if len({len(v) for v in values}) == 1 and len(values[0]) >= MIN_FIXED_CODE_LENGTH:
        return f"all values are {len(values[0])} digits long"
    if len(set(values)) / len(values) >= NEAR_UNIQUE_RATIO:
        return "values are almost all distinct"
    return None


def classify_column(
    name: str,
    raw_values: List[Optional[str]],
    categorical_max_unique: int = 50,
    categorical_max_ratio: float = 0.05,
    force_numeric: bool = False,
) -> ColumnClassification:
    non_null = [v for v in raw_values if not is_missing(v)]
    n_non_null = len(non_null)

    if n_non_null == 0:
        log.warning("%s: column is entirely missing/empty; defaulting to numeric/float", name)
        return ColumnClassification(dtype=NUMERIC, subtype="float", n_non_null=0)

    if not force_numeric:
        trimmed = [v.strip() for v in non_null]
        reason = integer_code_reason(trimmed)
        if reason:
            n_unique = len(set(trimmed))
            ratio = n_unique / n_non_null
            dtype = IDENTIFIER if ratio >= NEAR_UNIQUE_RATIO else CATEGORICAL
            log.warning(
                "%s: all-digit column (%s); classified as %s instead of numeric so "
                "real values are not reproduced. If this is a real number column, "
                "re-run profiling with --treat-as-number %s.",
                name, reason, dtype, name,
            )
            return ColumnClassification(
                dtype=dtype, n_non_null=n_non_null, n_unique=n_unique,
                cardinality_ratio=ratio, is_fully_unique=n_unique == n_non_null,
                non_null_values=trimmed,
            )

    numeric_hits = sum(1 for v in non_null if try_parse_number(v) is not None)
    if numeric_hits / n_non_null >= PARSE_THRESHOLD:
        if numeric_hits != n_non_null:
            log.warning(
                "%s: %d/%d non-null values failed to parse as numeric and will be "
                "treated as missing for stats purposes",
                name, n_non_null - numeric_hits, n_non_null,
            )
        parseable = [v for v in non_null if try_parse_number(v) is not None]
        subtype = "integer" if all(looks_like_int(v) for v in parseable) else "float"
        return ColumnClassification(
            dtype=NUMERIC, subtype=subtype, n_non_null=n_non_null,
            non_null_values=parseable,
        )

    date_hits = sum(1 for v in non_null if try_parse_date(v) is not None)
    if date_hits / n_non_null >= PARSE_THRESHOLD:
        if date_hits != n_non_null:
            log.warning(
                "%s: %d/%d non-null values failed to parse as date and will be "
                "treated as missing for stats purposes",
                name, n_non_null - date_hits, n_non_null,
            )
        parseable = [v for v in non_null if try_parse_date(v) is not None]
        return ColumnClassification(dtype=DATE, n_non_null=n_non_null, non_null_values=parseable)

    n_unique = len(set(non_null))
    cardinality_ratio = n_unique / n_non_null
    is_fully_unique = n_unique == n_non_null

    if n_unique <= categorical_max_unique or cardinality_ratio < categorical_max_ratio:
        return ColumnClassification(
            dtype=CATEGORICAL, n_non_null=n_non_null, n_unique=n_unique,
            cardinality_ratio=cardinality_ratio, is_fully_unique=is_fully_unique,
            non_null_values=non_null,
        )

    if is_fully_unique:
        log.warning(
            "%s: high-cardinality text column detected (%d unique / %d rows). "
            "Classified as identifier - review for PII before sharing profile.json.",
            name, n_unique, n_non_null,
        )
        return ColumnClassification(
            dtype=IDENTIFIER, n_non_null=n_non_null, n_unique=n_unique,
            cardinality_ratio=cardinality_ratio, is_fully_unique=True,
            non_null_values=non_null,
        )

    log.warning(
        "%s: high-cardinality text column detected (%d unique / %d rows). "
        "Classified as free_text - review, likely a name/free-text field with PII risk.",
        name, n_unique, n_non_null,
    )
    return ColumnClassification(
        dtype=FREE_TEXT, n_non_null=n_non_null, n_unique=n_unique,
        cardinality_ratio=cardinality_ratio, is_fully_unique=False,
        non_null_values=non_null,
    )

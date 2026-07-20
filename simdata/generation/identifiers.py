"""Pattern-based generator for identifier / free_text columns (Sec 6.2).

Never resamples real values (PII risk) - generates values matching the
profiled positional pattern (or, if no uniform pattern was found, a
length-bounded random string), enforcing uniqueness when the real column
was fully unique.
"""

from __future__ import annotations

import logging
import random
import string
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_CLASS_ALPHABETS = {
    "digit": string.digits,
    "upper": string.ascii_uppercase,
    "lower": string.ascii_lowercase,
    "alpha": string.ascii_letters,
    "any": string.ascii_letters + string.digits,
}

MAX_UNIQUE_RETRIES = 200


def _generate_one(template: Optional[List], length_min: int, length_max: int, rng: random.Random) -> str:
    if template is None:
        length = rng.randint(length_min, length_max)
        alphabet = string.ascii_letters + string.digits
        return "".join(rng.choice(alphabet) for _ in range(length))

    parts = []
    for item in template:
        if item[0] == "literal":
            parts.append(item[1])
        else:
            _, kind, run_len = item
            alphabet = _CLASS_ALPHABETS.get(kind, _CLASS_ALPHABETS["any"])
            parts.append("".join(rng.choice(alphabet) for _ in range(run_len)))
    return "".join(parts)


def generate(
    col_name: str,
    col_profile: Dict,
    n_records: int,
    rng: random.Random,
    enforce_unique: bool = False,
) -> List[str]:
    template = col_profile.get("pattern_template")
    length_stats = col_profile.get("length_stats", {"min": 8, "max": 8})
    length_min, length_max = length_stats["min"], length_stats["max"]

    values: List[str] = []
    seen = set() if enforce_unique else None
    collisions = 0
    forced_suffix = 0

    for _ in range(n_records):
        candidate = _generate_one(template, length_min, length_max, rng)
        if enforce_unique:
            attempts = 0
            while candidate in seen and attempts < MAX_UNIQUE_RETRIES:
                candidate = _generate_one(template, length_min, length_max, rng)
                attempts += 1
                collisions += 1
            if candidate in seen:
                # Pattern space exhausted relative to n_records - force
                # uniqueness with a suffix rather than looping forever.
                forced_suffix += 1
                candidate = f"{candidate}-{forced_suffix}"
            seen.add(candidate)
        values.append(candidate)

    if collisions:
        log.warning(
            "%s: %d uniqueness collisions during identifier generation (retried)",
            col_name, collisions,
        )
    if forced_suffix:
        log.warning(
            "%s: pattern space too small for %d unique records; %d values got a "
            "forced disambiguating suffix",
            col_name, n_records, forced_suffix,
        )

    return values

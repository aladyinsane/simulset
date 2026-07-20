"""Pattern/format inference for identifier and free-text columns (Sec 4, Sec 5.2)."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# Cap on how many values we inspect when inferring a positional pattern -
# it's a heuristic, and scanning every row of a 500K-row identifier column
# buys nothing beyond a few thousand samples.
MAX_SAMPLE = 3000

_CLASS_REGEX = {
    "digit": "[0-9]",
    "upper": "[A-Z]",
    "lower": "[a-z]",
    "alpha": "[A-Za-z]",
    "any": ".",
}


def _char_class(chars: set) -> str:
    if all(c.isdigit() for c in chars):
        return "digit"
    if all(c.isupper() for c in chars):
        return "upper"
    if all(c.islower() for c in chars):
        return "lower"
    if all(c.isalpha() for c in chars):
        return "alpha"
    return "any"


def infer_pattern(values: List[str]) -> Dict:
    """Infer a positional character-class pattern shared by all values.

    Returns a dict with a regex, a generator template (list of ("literal", str)
    or ("class", kind, run_length) tuples), and length stats. If values don't
    share a uniform length, only length stats are returned (uniform_length=False)
    and callers should fall back to length-bounded random generation.
    """
    lengths = [len(v) for v in values]
    length_min, length_max = min(lengths), max(lengths)

    result = {
        "regex": None,
        "template": None,
        "length_min": length_min,
        "length_max": length_max,
        "uniform_length": length_min == length_max,
    }

    if length_min != length_max:
        return result

    sample = values[:MAX_SAMPLE]
    length = length_min

    positions = []
    for i in range(length):
        chars_at_i = {v[i] for v in sample}
        if len(chars_at_i) == 1:
            positions.append(("literal", next(iter(chars_at_i))))
        else:
            positions.append((_char_class(chars_at_i), None))

    template = []
    regex_parts = []
    i = 0
    while i < len(positions):
        kind, lit = positions[i]
        j = i
        while j < len(positions) and positions[j][0] == kind and positions[j][1] == lit:
            j += 1
        run_len = j - i
        if kind == "literal":
            regex_parts.append(re.escape(lit) * run_len)
            template.append(("literal", lit * run_len))
        else:
            regex_parts.append(f"{_CLASS_REGEX[kind]}{{{run_len}}}")
            template.append(("class", kind, run_len))
        i = j

    result["regex"] = "^" + "".join(regex_parts) + "$"
    result["template"] = template
    return result

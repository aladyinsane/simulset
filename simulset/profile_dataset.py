"""Stage 1: profile a real (CSV-exported) dataset into a portable profile.json.

Usage:
    python -m simulset.profile_dataset --input real.csv --output profile.json \
        --log-file profile.log --log-level INFO
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .log_config import configure_logging
from .profiling import classify, stats

log = logging.getLogger(__name__)

PROFILER_VERSION = "1.0"


def read_csv_columns(
    path: str, encoding: str = "utf-8-sig"
) -> Tuple[List[str], Dict[str, List[Optional[str]]], int]:
    with open(path, newline="", encoding=encoding) as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path}: file is empty, no header row found")

        columns: Dict[str, List[Optional[str]]] = {h: [] for h in header}
        n_cols = len(header)
        row_count = 0
        blank_rows_skipped = 0

        for row in reader:
            if not row:
                blank_rows_skipped += 1
                continue
            row_count += 1
            for i, h in enumerate(header):
                columns[h].append(row[i] if i < len(row) else None)
            if len(row) > n_cols:
                log.debug("Row %d has %d fields, more than the %d header columns; extras dropped",
                          row_count, len(row), n_cols)

    if blank_rows_skipped:
        log.info("Skipped %d blank line(s) in %s", blank_rows_skipped, path)

    return header, columns, row_count


def profile_columns(
    header: List[str],
    columns: Dict[str, List[Optional[str]]],
    row_count: int,
    categorical_max_unique: int = 50,
    categorical_max_ratio: float = 0.05,
    treat_as_number: Optional[set] = None,
) -> Tuple[Dict[str, Dict], Optional[Dict]]:
    treat_as_number = treat_as_number or set()
    for c in sorted(treat_as_number - set(header)):
        log.warning("--treat-as-number column '%s' not found in the input header", c)
    profile: Dict[str, Dict] = {}
    numeric_aligned: Dict[str, List[Optional[float]]] = {}

    for name in header:
        raw_values = columns[name]
        cls = classify.classify_column(
            name, raw_values,
            categorical_max_unique=categorical_max_unique,
            categorical_max_ratio=categorical_max_ratio,
            force_numeric=name in treat_as_number,
        )
        missing_rate = 1.0 - (cls.n_non_null / row_count) if row_count else 0.0

        if cls.dtype == classify.EMPTY:
            col_stats = {"dtype": "empty"}
            missing_rate = 1.0
        elif cls.dtype == classify.NUMERIC:
            parsed = [classify.try_parse_number(v) for v in cls.non_null_values]
            col_stats = stats.numeric_stats(name, parsed, cls.non_null_values)
            col_stats["subtype"] = cls.subtype
            numeric_aligned[name] = [
                None if classify.is_missing(v) else classify.try_parse_number(v)
                for v in raw_values
            ]
        elif cls.dtype == classify.DATE:
            parsed_dates = [classify.try_parse_date(v) for v in cls.non_null_values]
            col_stats = stats.date_stats(parsed_dates)
        elif cls.dtype == classify.CATEGORICAL:
            col_stats = stats.categorical_stats(cls.non_null_values)
        elif cls.dtype == classify.IDENTIFIER:
            col_stats = stats.identifier_stats(
                name, cls.non_null_values, cls.n_unique, cls.is_fully_unique
            )
        else:  # FREE_TEXT
            col_stats = stats.free_text_stats(name, cls.non_null_values, cls.n_unique)

        col_stats["missing_rate"] = missing_rate
        profile[name] = col_stats

        log.info("%s: classified as %s (missing_rate=%.4f)", name, cls.dtype, missing_rate)
        if missing_rate > 0.5:
            log.warning("%s: high missing rate (%.1f%%)", name, missing_rate * 100)
        log.debug("%s: stats=%s", name, {k: v for k, v in col_stats.items() if k not in
                                          ("empirical_quantiles", "empirical_quantiles_epoch_day")})

    correlation_matrix = None
    if len(numeric_aligned) >= 2:
        correlation_matrix = stats.compute_correlation_matrix(numeric_aligned)

    return profile, correlation_matrix


def build_profile(
    input_path: str,
    categorical_max_unique: int,
    categorical_max_ratio: float,
    encoding: str = "utf-8-sig",
    treat_as_number: Optional[set] = None,
) -> Dict:
    header, columns, row_count = read_csv_columns(input_path, encoding)
    log.info("Read %d rows x %d columns from %s", row_count, len(header), input_path)

    columns_profile, correlation_matrix = profile_columns(
        header, columns, row_count, categorical_max_unique, categorical_max_ratio,
        treat_as_number,
    )

    profile = {
        "meta": {
            "source_file": input_path,
            "row_count": row_count,
            "column_order": header,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "profiler_version": PROFILER_VERSION,
        },
        "columns": columns_profile,
    }
    if correlation_matrix is not None:
        profile["correlation_matrix"] = correlation_matrix

    return profile


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a real dataset into profile.json")
    parser.add_argument("--input", required=True, help="Path to the real dataset CSV")
    parser.add_argument("--input-encoding", default="utf-8-sig",
                         help="Encoding of --input (default utf-8-sig). SAS PROC EXPORT's output "
                              "encoding matches the SAS session encoding, not utf-8 - check yours "
                              "(e.g. many Windows SAS installs default to cp1252) and pass it here "
                              "if it differs from the default")
    parser.add_argument("--output", required=True, help="Path to write profile.json")
    parser.add_argument("--categorical-max-unique", type=int, default=50,
                         help="Max distinct values for a string column to be categorical (default 50)")
    parser.add_argument("--categorical-max-ratio", type=float, default=0.05,
                         help="Max unique/non-null ratio for a string column to be categorical (default 0.05)")
    parser.add_argument("--treat-as-number", default="",
                         help="Comma-separated columns to keep numeric even if all their values are "
                              "digits (for a real number column the all-digit ID/code rule catches by mistake)")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file, args.log_level)
    log.info("profile_dataset starting: %s", vars(args))
    t0 = time.time()

    try:
        profile = build_profile(
            args.input, args.categorical_max_unique, args.categorical_max_ratio, args.input_encoding,
            {c.strip() for c in args.treat_as_number.split(",") if c.strip()},
        )
    except (OSError, ValueError) as exc:
        log.error("Failed to profile %s: %s", args.input, exc)
        return 1

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    elapsed = time.time() - t0
    log.info(
        "Profiled %d rows x %d columns from %s to %s in %.1fs",
        profile["meta"]["row_count"], len(profile["meta"]["column_order"]),
        args.input, args.output, elapsed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

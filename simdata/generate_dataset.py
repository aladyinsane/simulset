"""Stage 2: generate a synthetic CSV from profile.json (Sec 6).

Usage:
    python -m simdata.generate_dataset \
        --profile profile.json --n-records 100000 \
        --use-real-values color,state_code --real-csv real.csv \
        --preserve-correlations true --seed 42 \
        --output simulated.csv --log-file generate.log --log-level INFO

Note on --use-real-values and profile.json portability: profile.json embeds
full frequency tables for categorical columns, so flagging a categorical
column needs no further access to the real data. Every other dtype only has
summary statistics in profile.json, so flagging a numeric/date/identifier/
free_text column requires --real-csv to re-read its raw values for bootstrap
sampling (see Sec 6.1's table).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .generation import categorical, copula, dates, flagged, identifiers, missingness, numeric
from .log_config import configure_logging
from .profiling import classify

log = logging.getLogger(__name__)


def _str2bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def read_columns_from_csv(path: str, wanted_columns: List[str]) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {c: [] for c in wanted_columns}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing_cols = set(wanted_columns) - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"Columns not found in {path}: {sorted(missing_cols)}")
        for row in reader:
            for c in wanted_columns:
                v = row.get(c)
                if v is not None and not classify.is_missing(v):
                    result[c].append(v)
    return result


def format_numeric_column(raw_values: np.ndarray, subtype: str, decimals: int) -> List[str]:
    formatted = numeric.format_numeric(raw_values, subtype, decimals)
    if subtype == "integer":
        return [str(int(v)) for v in formatted]
    fmt = f"{{:.{decimals}f}}"
    return [fmt.format(v) for v in formatted]


def write_csv(path: str, column_order: List[str], generated: Dict[str, List[Optional[str]]], n_records: int) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(column_order)
        for i in range(n_records):
            writer.writerow(
                [generated[col][i] if generated[col][i] is not None else "" for col in column_order]
            )


def write_category_mapping(path: str, mapping_records: Dict[str, Dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["column", "real_value", "synthetic_value"])
        for col, mapping in mapping_records.items():
            for real_value, synthetic_value in mapping.items():
                writer.writerow([col, real_value, synthetic_value])


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a simulated dataset from profile.json")
    parser.add_argument("--profile", required=True, help="Path to profile.json from Stage 1")
    parser.add_argument("--n-records", required=True, type=int, help="Number of simulated rows to generate")
    parser.add_argument("--use-real-values", default="",
                         help="Comma-separated columns to bootstrap-sample literal real values for")
    parser.add_argument("--real-csv", default=None,
                         help="Path to the real CSV; required if a non-categorical column is in --use-real-values")
    parser.add_argument("--preserve-correlations", default="true",
                         help="true/false - use the Gaussian copula to preserve numeric cross-column correlation")
    parser.add_argument("--randomize-category-labels", action="store_true",
                         help="Shuffle synthetic category label order (still seeded by --seed)")
    parser.add_argument("--emit-category-mapping", action="store_true",
                         help="Write real->synthetic category label mapping to <output>_category_mapping.csv "
                              "(kept separate from the output file - real labels may be sensitive)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file, args.log_level)
    log.info("generate_dataset starting: %s", vars(args))
    t0 = time.time()

    preserve_correlations = _str2bool(args.preserve_correlations)

    try:
        with open(args.profile, encoding="utf-8") as f:
            profile = json.load(f)
    except OSError as exc:
        log.error("Failed to read profile %s: %s", args.profile, exc)
        return 1

    column_order = profile["meta"]["column_order"]
    columns_profile = profile["columns"]

    use_real_values = [c.strip() for c in args.use_real_values.split(",") if c.strip()]
    for c in use_real_values:
        if c not in columns_profile:
            log.error("--use-real-values column '%s' not found in profile", c)
            return 1
    use_real_values_set = set(use_real_values)

    needs_raw_csv = [c for c in use_real_values if columns_profile[c]["dtype"] != "categorical"]
    real_raw: Dict[str, List[str]] = {}
    if needs_raw_csv:
        if not args.real_csv:
            log.error(
                "Columns %s are flagged with --use-real-values but are not categorical, so "
                "their literal values must be re-read from the source CSV. Pass --real-csv.",
                needs_raw_csv,
            )
            return 1
        try:
            real_raw = read_columns_from_csv(args.real_csv, needs_raw_csv)
        except (OSError, ValueError) as exc:
            log.error("Failed to read --real-csv for flagged columns: %s", exc)
            return 1

    seed = args.seed
    np_rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    log.debug("Random seed: %s", seed)

    corr = profile.get("correlation_matrix")
    copula_columns: List[str] = []
    u_matrix: Optional[np.ndarray] = None
    if preserve_correlations and corr and len(corr.get("columns", [])) >= 2:
        copula_columns = [c for c in corr["columns"] if c not in use_real_values_set]
        if len(copula_columns) >= 2:
            idx = [corr["columns"].index(c) for c in copula_columns]
            sub_matrix = np.array(corr["matrix"], dtype=float)[np.ix_(idx, idx)]
            u_matrix = copula.draw_correlated_uniforms(args.n_records, sub_matrix, np_rng)
            log.info("Preserving correlations via Gaussian copula for columns: %s", copula_columns)
        else:
            copula_columns = []
            log.info(
                "Fewer than 2 non-flagged numeric columns available for the copula; "
                "generating numeric columns independently"
            )
    elif preserve_correlations:
        log.info("No usable correlation matrix in profile; generating numeric columns independently")
    else:
        log.info("--preserve-correlations=false; generating numeric columns independently")

    generated: Dict[str, List[Optional[str]]] = {}
    mapping_records: Dict[str, Dict[str, str]] = {}

    for col in column_order:
        col_profile = columns_profile[col]
        dtype = col_profile["dtype"]

        if col in use_real_values_set:
            if dtype == "categorical":
                raw_choice = flagged.generate_categorical_flagged(col_profile, args.n_records, np_rng)
                values: List[Optional[str]] = [str(v) for v in raw_choice]
            else:
                values = flagged.generate_from_raw_values(real_raw[col], args.n_records, py_rng)
            strategy = "real-value-passthrough"

        elif dtype == "numeric":
            if col in copula_columns:
                u = u_matrix[:, copula_columns.index(col)]
                strategy = "empirical-cdf+copula"
            else:
                u = np_rng.random(args.n_records)
                strategy = "empirical-cdf"
            raw_vals = numeric.empirical_cdf_sample(col_profile["empirical_quantiles"], u)
            values = format_numeric_column(raw_vals, col_profile["subtype"], col_profile["decimals"])

        elif dtype == "categorical":
            arr, mapping = categorical.generate(
                col, col_profile, args.n_records, np_rng, args.randomize_category_labels
            )
            values = [str(v) for v in arr]
            mapping_records[col] = mapping
            strategy = "synthetic-label"

        elif dtype == "date":
            u = np_rng.random(args.n_records)
            values = dates.generate(col_profile, u)
            strategy = "empirical-cdf"

        elif dtype == "identifier":
            values = identifiers.generate(
                col, col_profile, args.n_records, py_rng,
                enforce_unique=col_profile.get("is_fully_unique", False),
            )
            strategy = "pattern-generated"

        else:  # free_text
            values = identifiers.generate(col, col_profile, args.n_records, py_rng, enforce_unique=False)
            strategy = "pattern-generated"

        log.info("%s: strategy=%s", col, strategy)
        log.debug("%s: sample=%s", col, values[:5])

        values = missingness.inject(values, col_profile.get("missing_rate", 0.0), py_rng)
        generated[col] = values

    write_csv(args.output, column_order, generated, args.n_records)

    if args.emit_category_mapping and mapping_records:
        mapping_path = str(Path(args.output).with_name(Path(args.output).stem + "_category_mapping.csv"))
        write_category_mapping(mapping_path, mapping_records)
        log.info(
            "Wrote category label mapping to %s - keep separate from output, real labels may be sensitive",
            mapping_path,
        )

    elapsed = time.time() - t0
    log.info(
        "Generated %d records x %d columns to %s in %.1fs (seed=%s)",
        args.n_records, len(column_order), args.output, elapsed, seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

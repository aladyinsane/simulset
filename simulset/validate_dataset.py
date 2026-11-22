"""Stage 3 (optional): re-profile simulated.csv and diff it against the
original profile.json (Sec 7) - real vs. simulated summary stats per column
and a correlation-matrix diff, which is the main check that Sec 6.3's
copula step actually worked.

Usage:
    python -m simulset.validate_dataset --profile profile.json \
        --simulated simulated.csv --report report.json \
        --log-file validate.log --log-level INFO

Leak check (add --real-csv): counts, per column, how many simulated values
equal a real value. Only counts are ever printed, logged or reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from .log_config import configure_logging
from .generate_dataset import read_columns_from_csv
from .profile_dataset import build_profile

log = logging.getLogger(__name__)

NUMERIC_FIELDS = ["min", "max", "mean", "median", "variance"]
DATE_FIELDS = ["min", "max"]
CARDINALITY_FIELDS = ["n_unique"]
# Fields that are literal data values. With --real-csv the output may be pasted
# anywhere, so these are left out of the stats diff.
LITERAL_VALUE_FIELDS = {"min", "max", "median"}
LEAK_CHECK_DTYPES = ("identifier", "free_text", "categorical")


def _field_diff(real_col: Dict, sim_col: Dict, field: str) -> Dict:
    real_v = real_col.get(field)
    sim_v = sim_col.get(field)
    entry = {"real": real_v, "simulated": sim_v}
    if isinstance(real_v, (int, float)) and isinstance(sim_v, (int, float)):
        entry["abs_diff"] = abs(real_v - sim_v)
    return entry


def diff_column(name: str, real_col: Dict, sim_col: Optional[Dict], redact_values: bool = False) -> Dict:
    if sim_col is None:
        return {"error": "column missing from simulated dataset"}

    diff = {
        "real_dtype": real_col["dtype"],
        "simulated_dtype": sim_col["dtype"],
        "missing_rate": _field_diff(real_col, sim_col, "missing_rate"),
    }
    if real_col["dtype"] != sim_col["dtype"]:
        log.warning(
            "%s: dtype classification changed between real (%s) and simulated (%s)",
            name, real_col["dtype"], sim_col["dtype"],
        )

    if real_col["dtype"] == "numeric" and sim_col["dtype"] == "numeric":
        for field in NUMERIC_FIELDS:
            if redact_values and field in LITERAL_VALUE_FIELDS:
                continue
            diff[field] = _field_diff(real_col, sim_col, field)
    elif real_col["dtype"] == "date" and sim_col["dtype"] == "date":
        for field in DATE_FIELDS:
            if redact_values and field in LITERAL_VALUE_FIELDS:
                continue
            diff[field] = _field_diff(real_col, sim_col, field)
    elif real_col["dtype"] == "categorical" and sim_col["dtype"] == "categorical":
        diff["n_unique"] = _field_diff(real_col, sim_col, "n_unique")
        real_top = sorted(real_col["frequencies"].values(), reverse=True)[:5]
        sim_top = sorted(sim_col["frequencies"].values(), reverse=True)[:5]
        diff["top_frequencies_by_rank"] = {"real": real_top, "simulated": sim_top}
    elif real_col["dtype"] in ("identifier", "free_text"):
        diff["n_unique"] = _field_diff(real_col, sim_col, "n_unique")
        diff["length_stats"] = {
            "real": real_col.get("length_stats"), "simulated": sim_col.get("length_stats"),
        }

    return diff


def diff_correlation_matrix(real_corr: Optional[Dict], sim_corr: Optional[Dict]) -> Optional[Dict]:
    if real_corr is None:
        return None
    if sim_corr is None:
        log.warning("Real profile had a correlation matrix but the simulated dataset did not produce one")
        return {"error": "simulated dataset has no correlation matrix"}

    common_cols = [c for c in real_corr["columns"] if c in sim_corr["columns"]]
    if len(common_cols) < 2:
        log.warning("Fewer than 2 columns in common between real and simulated correlation matrices")
        return {"error": "fewer than 2 common columns"}

    real_idx = [real_corr["columns"].index(c) for c in common_cols]
    sim_idx = [sim_corr["columns"].index(c) for c in common_cols]
    real_m = np.array(real_corr["matrix"])[np.ix_(real_idx, real_idx)]
    sim_m = np.array(sim_corr["matrix"])[np.ix_(sim_idx, sim_idx)]

    abs_diff = np.abs(real_m - sim_m)
    return {
        "columns": common_cols,
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "frobenius_norm_diff": float(np.linalg.norm(real_m - sim_m)),
    }


def build_report(real_profile: Dict, sim_profile: Dict, redact_values: bool = False) -> Dict:
    real_columns = real_profile["columns"]
    sim_columns = sim_profile["columns"]

    columns_report = {}
    for name in real_profile["meta"]["column_order"]:
        columns_report[name] = diff_column(
            name, real_columns[name], sim_columns.get(name), redact_values
        )

    return {
        "row_counts": {
            "real": real_profile["meta"]["row_count"],
            "simulated": sim_profile["meta"]["row_count"],
        },
        "columns": columns_report,
        "correlation_matrix": diff_correlation_matrix(
            real_profile.get("correlation_matrix"), sim_profile.get("correlation_matrix")
        ),
    }


def read_header(path: str, encoding: str = "utf-8-sig") -> List[str]:
    with open(path, newline="", encoding=encoding) as f:
        try:
            return next(csv.reader(f))
        except StopIteration:
            raise ValueError(f"{path}: file is empty, no header row found")


def match_keys(value: str) -> List[str]:
    """Forms a value is compared in: trimmed and case-folded; all-digit values
    also with leading zeros stripped, so 12345678 matches 012345678."""
    v = value.strip().casefold()
    if v.isascii() and v.isdigit():
        return [v, v.lstrip("0") or "0"]
    return [v]


NOT_CHECKED = "not in real.csv, not checked"


def run_leak_check(
    real_profile: Dict,
    real_csv: str,
    simulated_csv: str,
    encoding: str,
    allow: List[str],
    extra_columns: List[str],
) -> Dict:
    """Count simulated values equal to a real value, per column. Returns only
    counts. Raises ValueError for unknown columns.

    Checked: profile columns that are identifier, free_text or categorical;
    named columns; and any column in both CSVs that the profile lacks. A
    simulated column the profile lacks and real.csv lacks cannot be checked, so
    it fails unless allowed. Matching ignores case and surrounding spaces, and
    leading zeros on all-digit values."""
    profile_cols = real_profile["columns"]
    real_header = read_header(real_csv, encoding)
    sim_header = read_header(simulated_csv)

    unknown_extra = [
        c for c in extra_columns
        if c not in profile_cols and not (c in real_header and c in sim_header)
    ]
    if unknown_extra:
        raise ValueError(
            f"--leak-check-columns not in the profile or not in both CSVs: {sorted(set(unknown_extra))}"
        )
    unknown_allow = [c for c in allow if c not in profile_cols and c not in sim_header]
    if unknown_allow:
        raise ValueError(f"--allow-real-values not found in the profile or simulated file: {sorted(set(unknown_allow))}")

    to_check: List[str] = []
    for c in real_profile["meta"]["column_order"]:
        if profile_cols[c]["dtype"] in LEAK_CHECK_DTYPES or c in extra_columns:
            to_check.append(c)
    for c in sim_header:  # columns the profile does not know about
        if c not in profile_cols and c in real_header and c not in to_check:
            to_check.append(c)
    unchecked = [c for c in sim_header if c not in profile_cols and c not in real_header]

    real_raw = read_columns_from_csv(real_csv, to_check, encoding)
    sim_raw = read_columns_from_csv(simulated_csv, to_check)

    columns: Dict[str, Dict] = {}
    for c in to_check:
        real_keys = set()
        for v in real_raw[c]:
            real_keys.update(match_keys(v))
        matching = sum(1 for v in sim_raw[c] if any(k in real_keys for k in match_keys(v)))
        columns[c] = {
            "dtype": profile_cols[c]["dtype"] if c in profile_cols else "unprofiled",
            "checked": len(sim_raw[c]),
            "matching": matching,
            "allowed": c in allow,
        }
    for c in unchecked:
        columns[c] = {"dtype": "unprofiled", "checked": 0, "matching": 0,
                      "allowed": c in allow, "status": NOT_CHECKED}

    failing = [
        c for c, e in columns.items()
        if not e["allowed"] and (e["matching"] > 0 or e.get("status") == NOT_CHECKED)
    ]
    return {"columns": columns, "failing_columns": len(failing), "passed": not failing}


def print_leak_check(leak: Dict) -> None:
    print()
    print("Leak check (counts only, no values shown)")
    header = f"{'column':30} {'dtype':12} {'checked':>10} {'matching':>10} {'allowed':>8}"
    print(header)
    print("-" * len(header))
    for name, e in leak["columns"].items():
        note = f"  {e['status']}" if "status" in e else ""
        print(f"{name:30} {e['dtype']:12} {e['checked']:>10} {e['matching']:>10} {str(e['allowed']):>8}{note}")
        log.info("leak check %s: checked=%d matching=%d allowed=%s%s",
                 name, e["checked"], e["matching"], e["allowed"], note)


def print_report(report: Dict) -> None:
    print(f"Row count: real={report['row_counts']['real']} simulated={report['row_counts']['simulated']}")
    print()
    header = f"{'column':30} {'dtype':12} {'field':10} {'real':>14} {'simulated':>14}"
    print(header)
    print("-" * len(header))
    for name, col_diff in report["columns"].items():
        if "error" in col_diff:
            print(f"{name:30} {'':12} {'ERROR':10} {col_diff['error']}")
            continue
        dtype = col_diff["real_dtype"]
        for field, values in col_diff.items():
            if field in ("real_dtype", "simulated_dtype") or not isinstance(values, dict):
                continue
            if "real" in values and "simulated" in values:
                real_v = values["real"]
                sim_v = values["simulated"]
                print(f"{name:30} {dtype:12} {field:10} {str(real_v):>14} {str(sim_v):>14}")
    print()
    corr = report.get("correlation_matrix")
    if corr and "error" not in corr:
        print(
            f"Correlation matrix diff ({len(corr['columns'])} common columns): "
            f"max_abs_diff={corr['max_abs_diff']:.4f} "
            f"mean_abs_diff={corr['mean_abs_diff']:.4f} "
            f"frobenius_norm_diff={corr['frobenius_norm_diff']:.4f}"
        )
    elif corr:
        print(f"Correlation matrix diff: {corr['error']}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a simulated dataset against its profile.json")
    parser.add_argument("--profile", required=True, help="Path to the real profile.json from Stage 1")
    parser.add_argument("--simulated", required=True, help="Path to the simulated CSV from Stage 2")
    parser.add_argument("--report", default=None, help="Optional path to write the diff report as JSON")
    parser.add_argument("--real-csv", default=None,
                         help="Path to the real CSV. Turns on the leak check: counts simulated values "
                              "that equal a real value in identifier, free_text and categorical columns "
                              "(and --leak-check-columns). Only counts are shown, never values. "
                              "Exits 1 if any checked column not in --allow-real-values has a match.")
    parser.add_argument("--input-encoding", default="utf-8-sig",
                         help="Encoding of --real-csv (default utf-8-sig), same as profile/generate")
    parser.add_argument("--allow-real-values", default="",
                         help="Comma-separated columns where matching real values is expected "
                              "(for example columns passed to generate --use-real-values)")
    parser.add_argument("--leak-check-columns", default="",
                         help="Comma-separated extra columns (for example a numeric one) to leak-check")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file, args.log_level)
    log.info("validate_dataset starting: %s", vars(args))
    t0 = time.time()

    try:
        with open(args.profile, encoding="utf-8") as f:
            real_profile = json.load(f)
    except OSError as exc:
        log.error("Failed to read %s: %s", args.profile, exc)
        return 1

    allow = [c.strip() for c in args.allow_real_values.split(",") if c.strip()]
    extra = [c.strip() for c in args.leak_check_columns.split(",") if c.strip()]
    if (allow or extra) and not args.real_csv:
        log.error("--allow-real-values and --leak-check-columns need --real-csv")
        return 1

    leak = None
    if args.real_csv:
        try:
            leak = run_leak_check(
                real_profile, args.real_csv, args.simulated, args.input_encoding, allow, extra
            )
        except (OSError, ValueError, UnicodeError) as exc:
            log.error("Leak check failed: %s", exc)
            return 1
        # The profiler's DEBUG stats line can hold literal values (min/max, labels);
        # keep it out of the log when the simulated file may contain real ones.
        logging.getLogger("simulset.profile_dataset").setLevel(logging.INFO)

    try:
        sim_profile = build_profile(args.simulated, categorical_max_unique=50, categorical_max_ratio=0.05)
    except (OSError, ValueError) as exc:
        log.error("Failed to profile %s: %s", args.simulated, exc)
        return 1
    finally:
        logging.getLogger("simulset.profile_dataset").setLevel(logging.NOTSET)

    report = build_report(real_profile, sim_profile, redact_values=leak is not None)
    print_report(report)
    if leak is not None:
        report["leak_check"] = leak
        print_leak_check(leak)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        log.info("Wrote validation report to %s", args.report)

    elapsed = time.time() - t0
    log.info("Validated %s against %s in %.1fs", args.simulated, args.profile, elapsed)
    if leak is None:
        return 0
    verdict = (
        "LEAK CHECK: PASS" if leak["passed"]
        else f"LEAK CHECK: FAIL ({leak['failing_columns']} columns)"
    )
    log.info(verdict)
    print(verdict)  # last line of output
    return 0 if leak["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

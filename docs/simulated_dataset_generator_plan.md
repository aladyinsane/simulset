# Simulated Dataset Generator — Implementation Plan

## 1. Goal

Build a two-stage Python tool that reads a real dataset (originally SAS, exported to CSV)
and produces a synthetic CSV of a user-specified number of records, where each column
reproduces the statistical "shape" of the real column — and, for columns the user
explicitly names, reproduces the real values themselves in their real proportions.

## 2. Requirements (as given)

- Input: a real dataset, currently in SAS (`.sas7bdat`); CSV export is acceptable.
- Output: a CSV of simulated records.
- User specifies `n_records` for the output.
- Per column, the simulated data should match:
  - proportion of missing values
  - min, max, mean, median, variance
- User can supply a list of columns for which **actual real values are reused**, in the
  same proportions they appear in the real data (e.g., `color` with `orange/red/green`).
- Environment: Python with standard library only; pandas allowed only if truly necessary.
  Also available: SAS 9.4, cmd, PowerShell.
- Good logging throughout.
- Claude was asked to propose additional properties worth matching — see Section 4.

**Decisions confirmed (supersede any "open question" language below):**
- Non-flagged categorical columns: create a **new synthetic label for each distinct real
  category value** (a 1:1 mapping, real value → synthetic value), then sample those
  synthetic labels in the real proportions. Real category *values* are never written to
  output unless the column is explicitly on `--use-real-values`. See Sec 6.2.
- Cross-column correlation preservation (Sec 6.3) is **in scope for Phase 1**, not deferred.
- The real dataset is **large** (500K+ rows, possibly 100+ columns) — see the
  performance note in Sec 13, which recommends allowing `numpy` for the numeric/linear
  algebra pieces even though the rest of the pipeline stays stdlib-only.

## 3. Pipeline Overview

```
Stage 0 (SAS)      Stage 1 (Python)         Stage 2 (Python)        Stage 3 (Python, optional)
sas7bdat  --------> profile_dataset.py ----> generate_dataset.py --> validate_dataset.py
  |  PROC EXPORT         |                        |                       |
  v                      v                        v                       v
real.csv            profile.json  +  CLI args -> simulated.csv    comparison_report
```

Splitting profiling from generation is deliberate:
- `profile.json` is a portable, inspectable summary of the real data (no need to keep
  re-reading the real dataset every time you regenerate).
- You (or a reviewer) can eyeball `profile.json` before anything is simulated — useful for
  catching PII-bearing high-cardinality columns.
- `generate_dataset.py` can run repeatedly with different `n_records` or random seeds
  without touching the real data again.

## 4. Column-Level Properties to Capture

Beyond min/max/mean/median/variance/missing-rate (as requested), profile these per column:

| Category | Property | Why it matters |
|---|---|---|
| All columns | dtype (int/float/string/bool/date) | output must round-trip to the same type |
| All columns | missing-value proportion | requested |
| All columns | distinct-value count / cardinality ratio | drives categorical vs. free-text handling |
| Numeric | min, max, mean, median, variance/std | requested |
| Numeric | 1st/5th/25th/75th/95th/99th percentiles | preserves shape beyond mean/var (skew, tails) |
| Numeric | skewness, kurtosis | flags non-normal shapes so simulation doesn't look "too clean" |
| Numeric | decimal precision (e.g., 2 dp) / whether values are integers | avoids `19.999999` showing up where real data is `20.00` |
| Numeric | outlier presence (e.g., values beyond 1.5×IQR) | empirical-resampling approach (Sec 6.2) preserves these naturally, but worth logging |
| Categorical/string | frequency table of each category | requested (proportions) |
| Categorical/string | string length distribution | for free-text/ID-like columns |
| Categorical/string | value pattern (e.g., always 5 digits, leading zeros, fixed prefix) | so ZIP-code-like columns don't silently become integers |
| Date/time | min, max, weekday/month distribution | dates often have seasonality that a plain min/max sampler would erase |
| ID-like columns | uniqueness constraint | if the real column is 100% unique, the simulated one probably should be too |
| Cross-column | correlation matrix of numeric columns | independently-simulated numeric columns will look unrealistic (e.g., height/weight not moving together) — see Sec 6.3 |
| Cross-column | co-missingness (do columns tend to be missing together?) | stretch goal, flagged in Sec 10 |
| Whole dataset | row count, column order, column names | for reference/logging only |

## 5. `profile_dataset.py` — Stage 1

### 5.1 Column classification (decision logic)

```
for each column:
    if dtype is numeric (int/float):
        classify as NUMERIC
    elif dtype is datetime:
        classify as DATETIME
    else:  # string/object
        cardinality_ratio = n_unique / n_non_null
        if n_unique <= 50 OR cardinality_ratio < 0.05:
            classify as CATEGORICAL
        else:
            classify as FREE_TEXT / IDENTIFIER
            log.warning(f"{col}: high-cardinality text column detected "
                        f"({n_unique} unique / {n_non_null} rows). "
                        f"Review — likely a name/ID/free-text field.")
```

`FREE_TEXT` columns are logged loudly because they're the most likely place to
accidentally leak PII into a "simulated" dataset if just resampled by default.

### 5.2 `profile.json` schema (example)

```json
{
  "meta": {
    "source_file": "real.csv",
    "row_count": 48213,
    "column_order": ["patient_id", "age", "color", "enrollment_date", "cost"],
    "generated_at": "2026-07-19T10:04:00",
    "profiler_version": "1.0"
  },
  "correlation_matrix": {
    "columns": ["age", "cost"],
    "matrix": [[1.0, 0.42], [0.42, 1.0]]
  },
  "columns": {
    "age": {
      "dtype": "numeric",
      "subtype": "integer",
      "missing_rate": 0.012,
      "min": 0, "max": 97, "mean": 41.3, "median": 40, "variance": 210.5,
      "percentiles": {"p1": 2, "p5": 8, "p25": 27, "p75": 55, "p95": 78, "p99": 90},
      "skewness": 0.11, "kurtosis": -0.4,
      "decimals": 0
    },
    "color": {
      "dtype": "categorical",
      "missing_rate": 0.003,
      "n_unique": 3,
      "frequencies": {"orange": 0.51, "red": 0.33, "green": 0.16}
    },
    "enrollment_date": {
      "dtype": "date",
      "missing_rate": 0.0,
      "min": "2019-01-04", "max": "2026-06-30",
      "weekday_distribution": {"Mon": 0.24, "Tue": 0.22, "...": "..."}
    },
    "patient_id": {
      "dtype": "identifier",
      "missing_rate": 0.0,
      "n_unique": 48213,
      "is_fully_unique": true,
      "value_pattern": "^P[0-9]{6}$",
      "length_stats": {"min": 7, "max": 7}
    }
  }
}
```

## 6. `generate_dataset.py` — Stage 2

### 6.1 CLI parameters

```
python generate_dataset.py \
  --profile profile.json \
  --n-records 100000 \
  --use-real-values color,state_code \
  --preserve-correlations true \
  --seed 42 \
  --output simulated.csv \
  --log-file generate.log \
  --log-level INFO
```

| Arg | Purpose |
|---|---|
| `--profile` | path to `profile.json` from Stage 1 |
| `--n-records` | requested row count |
| `--use-real-values` | comma-separated list of columns to sample literal real values from (proportionally) — **requires the real values to still be reachable**, either embedded in `profile.json` as full frequency tables (fine for categorical) or read from real.csv again at generation time (needed if a *numeric* column is flagged, since literal-value bootstrapping needs the raw values, not just summary stats) |
| `--preserve-correlations` | on/off toggle for the copula step (Sec 6.3) |
| `--seed` | reproducibility |
| `--output` | output CSV path |
| `--log-file`, `--log-level` | logging controls |

### 6.2 Per-column generation strategy

**Flagged columns (`--use-real-values`):**
Bootstrap-sample directly from the real values in their observed proportions
(`random.choices(real_values, weights=frequencies, k=n_records)`). This applies whether
the column is numeric or categorical.

**Non-flagged NUMERIC columns:**
Use **empirical-CDF interpolation** (inverse-transform sampling against the sorted real
values), *not* a parametric distribution fit:
1. Sort the real column's non-null values.
2. For each simulated record, draw `u ~ Uniform(0,1)` and linearly interpolate between
   the two sorted real values whose percentile ranks bracket `u`.
3. Round to the profiled decimal precision; cast to int if the source was integer.

This is simple (pure Python, no distribution-fitting library needed), and by
construction reproduces min/max/median/percentiles/skew reasonably well without ever
literally reusing whole real records — better for the "simulated, not copied" intent
implied by only whitelisting *some* columns as real-value passthroughs. It handles
multimodal / skewed data much better than assuming normality.

**Non-flagged CATEGORICAL columns (confirmed default):**
Never write real category labels for these columns. Instead:
1. Take the profiled frequency table for the column (real label → proportion).
2. Generate one **synthetic replacement label per distinct real value** — default naming
   scheme: order real values by descending frequency and assign `f"{col}_val{i}"`
   (e.g., `color_val1`, `color_val2`, `color_val3`). This mapping is deterministic given
   the same `profile.json`, so reruns are reproducible without persisting the mapping —
   but also expose a `--randomize-category-labels` flag (seeded by `--seed`) if you'd
   rather the synthetic label *order* be shuffled per run.
3. Sample from the synthetic labels using the real proportions
   (`random.choices(synthetic_labels, weights=frequencies, k=n_records)`).
4. Optionally write the real→synthetic mapping to a separate `*_category_mapping.csv`
   (via `--emit-category-mapping`) for your own QA — keep this file separate from the
   output, since (depending on the dataset) the *real* label names themselves could be
   sensitive even though the simulated file no longer contains them.

This is the same principle already used for numeric columns (Sec 6.2 above) and
identifier columns (below): **flagged columns reuse real values verbatim; everything
else is synthesized to match real statistical properties without ever exposing a real
value.**

**FREE_TEXT / IDENTIFIER columns:**
Never resample real values by default (PII risk). Default behavior:
generate values matching the profiled pattern/length (e.g., `P` + 6 random digits for
`^P[0-9]{6}$`), and if `is_fully_unique` was true, enforce uniqueness in the output too
(regenerate on collision). Log a summary of how many such columns were auto-patterned so
you can spot-check them.

**DATE columns:**
Empirical-CDF interpolation on the epoch-day representation (same mechanism as numeric),
optionally re-weighted so the simulated weekday/month distribution matches the profiled
one.

### 6.3 Preserving cross-column correlation (numeric columns) — in scope for Phase 1

Independently sampling each numeric column (Sec 6.2) will destroy any real relationship
between columns (e.g., age vs. cost). To preserve rank correlation while keeping each
column's own marginal distribution intact, use a **Gaussian copula**:

1. Rank-transform each real numeric column to standard-normal scores.
2. Compute the correlation matrix of those normal scores (already in `profile.json`).
3. Compute a Cholesky decomposition of that correlation matrix.
4. Draw correlated standard-normal vectors (independent draws × Cholesky factor),
   convert back to uniform via the normal CDF, then feed each column's `u` through its
   own empirical-CDF interpolation (Sec 6.2) instead of sampling `u` independently per
   column.

**Performance note given the confirmed dataset scale (500K+ rows, possibly 100+
columns):** steps 1–3 are cheap regardless (rank-transform/sort is at worst tens of
seconds across 100 columns of 500K rows; Cholesky on a 100×100 matrix is trivial). Step
4, however, is a matrix–vector multiply repeated once per output row — at 100 columns ×
500K+ rows that's on the order of 5 billion scalar multiply-adds, which pure-Python loops
would handle too slowly to be practical. **Recommendation: use `numpy` for step 4 (and
for holding numeric columns generally)**, while keeping CSV I/O, categorical/identifier
handling, and logging in the standard library as planned. This is a narrow, justified
exception to the "stdlib only" preference — flag it if you'd rather explore a pure-Python
optimization (e.g., chunked processing) instead of adding the numpy dependency.

### 6.4 Missing-value injection

For each column independently, after generating `n_records` values, randomly null out
`missing_rate × n_records` of them (`random.sample` of row indices). Note this treats
missingness per-column as independent; true joint missingness patterns are a stretch
goal (Sec 10).

### 6.5 Output formatting

- Preserve column order from `profile.json["meta"]["column_order"]`.
- Cast each column back to its profiled dtype/precision before writing.
- Write missing values as empty strings (or `NA` — confirm CSV convention your
  downstream tools expect).

## 7. `validate_dataset.py` — Stage 3 (recommended, optional)

After generation, recompute the same profile on `simulated.csv` and print/log a
side-by-side diff against `profile.json`: real vs. simulated mean/median/variance/
min/max/missing-rate per column, and real vs. simulated correlation matrix. This is
cheap to build (reuses Stage 1's profiling code) and is the main way you'll catch bugs
(e.g., a column accidentally shifted, a correlation not actually preserved).

## 8. Logging Plan

Use the standard `logging` module in all three scripts, configured once via a shared
helper (`log_config.py`):

- Handlers: console (INFO+) and rotating file handler (DEBUG+), path set by `--log-file`.
- Format: `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`.
- Log at minimum:
  - **INFO**: script start/end with args, row/column counts read, output file path and
    row count written, elapsed time per stage.
  - **INFO** per column processed: dtype classification decided, strategy chosen
    (real-value passthrough / empirical-CDF / pattern-generated).
  - **WARNING**: high-cardinality text columns detected (possible PII), columns with
    >X% missing, uniqueness collisions during identifier generation and how many retries
    it took, any column where correlation preservation was skipped (e.g., all-null or
    constant column).
  - **ERROR**: unreadable column, `--use-real-values` column not found in profile,
    n_records requested with no viable generation strategy.
  - **DEBUG**: full per-column stats dump, random seed used, sample of first 5 generated
    values per column.
- Every run should log a final one-line summary, e.g.:
  `Generated 100000 records x 5 columns to simulated.csv in 4.2s (seed=42)`.

## 9. Suggested File Structure

```
simulset/
  log_config.py          # shared logging setup
  profile_dataset.py      # Stage 1 CLI entry point
  generate_dataset.py     # Stage 2 CLI entry point
  validate_dataset.py     # Stage 3 CLI entry point
  profiling/
    classify.py           # column type/cardinality classification
    stats.py               # min/max/mean/median/variance/percentiles/skew/kurtosis
    patterns.py            # regex/format pattern inference for identifier columns
  generation/
    numeric.py             # empirical-CDF interpolation sampler
    categorical.py          # frequency-weighted sampler
    identifiers.py          # pattern-based + uniqueness-enforced generator
    copula.py                # numpy-backed Cholesky + Gaussian copula (Phase 1)
    missingness.py            # per-column missing-value injector
  sas/
    export_to_csv.sas       # PROC EXPORT script (Stage 0)
  tests/
    ...
```

## 10. Implementation Phases

**Phase 1 (MVP)** — stdlib + numpy (numpy scoped narrowly to numeric arrays / the
correlation-copula step, per Sec 6.3's performance note):
- `profile_dataset.py` producing full `profile.json` (Sec 5), including the numeric
  correlation matrix.
- `generate_dataset.py`: real-value passthrough columns, empirical-CDF numeric sampling,
  **Gaussian copula correlation preservation (Sec 6.3)**, synthetic-label categorical
  sampling (Sec 6.2), pattern-based identifier generation, independent per-column
  missingness.
- Logging as specified in Sec 8.
- `validate_dataset.py` comparison report, including real-vs-simulated correlation
  matrix diff (this is your main check that Sec 6.3 actually worked).

**Phase 2** — realism enhancements:
- Joint/co-missingness modeling.
- Date seasonality (weekday/month re-weighting).

**Phase 3** — nice-to-haves:
- Config file (YAML/JSON) as an alternative to long CLI arg lists.
- Per-column strategy overrides (force a specific numeric column to pattern-based, etc.).
- Support for user-supplied logical constraints (e.g., `end_date >= start_date`),
  enforced as a post-generation repair pass.

## 11. Stage 0: SAS export snippet

```sas
libname src "/path/to/sas/library";

proc export data=src.your_dataset
    outfile="/path/to/real.csv"
    dbms=csv
    replace;
run;
```

Run this once in SAS 9.4 to produce `real.csv`, which is then the sole input to the
Python pipeline. Keeping this step in SAS (rather than reading `.sas7bdat` from Python)
avoids needing a sas7bdat-reading dependency and sidesteps SAS format/informat quirks.

## 12. Open Questions for You to Confirm

Resolved: default categorical handling, correlation preservation timing, and dataset
scale are all settled — see the "Decisions confirmed" box near the top. Remaining:

1. **Synthetic category label naming.** Section 6.2 assumes a generic scheme
   (`color_val1`, `color_val2`, ...) ordered by descending real frequency. Fine as a
   default, or would you prefer something else (e.g., format-mimicking fake strings,
   or labels drawn from a fixed placeholder word list instead of `{col}_valN`)?
2. **Domain-specific constraints.** Are there known logical rules between columns
   (e.g., a date column that must stay after another date column, or a code that
   implies a category) that statistical matching alone won't capture? These would need
   to be handled as an explicit rules list (Phase 3).

## 13. Dependencies

Given the confirmed scale (500K+ rows, possibly 100+ columns):

- **stdlib** (`csv`, `statistics`, `random`, `json`, `argparse`, `logging`, `re`,
  `datetime`, `math`, `bisect`) for: CSV I/O, column classification, categorical/
  identifier/free-text generation, missingness injection, logging, and the overall
  CLI/orchestration.
- **numpy** — recommended, scoped narrowly to: storing numeric columns as arrays
  (memory efficiency at this row count), the correlation matrix computation, the
  Cholesky decomposition, and the copula's matrix–vector multiply (Sec 6.3). This is
  the one deliberate exception to "stdlib only," justified purely by the scale of the
  correlation-preserving step — everything else stays stdlib.
- **pandas** — still not required. If profiling throughput on the full CSV turns out to
  be a bottleneck even with numpy for the numeric side, `pandas.read_csv` is the next
  lever to pull, but start without it.
- Consider processing the CSV in **chunks** (`csv.reader` streaming, or numpy arrays
  built incrementally) rather than loading 500K rows × 100+ columns fully into memory
  at once, especially during profiling.

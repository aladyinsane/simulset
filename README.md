# simulset

Two-stage tool that profiles a real dataset and generates a synthetic CSV of
arbitrary size that reproduces each column's statistical shape - and, for
columns you name explicitly, the real values themselves in their real
proportions. Implements Phase 1 (MVP) of
[docs/simulated_dataset_generator_plan.md](docs/simulated_dataset_generator_plan.md).

## Pipeline

```
Stage 0 (SAS)        Stage 1                  Stage 2                  Stage 3 (optional)
sas7bdat  --------->  profile_dataset.py  -->  generate_dataset.py  -->  validate_dataset.py
   |  PROC EXPORT           |                         |                         |
   v                        v                         v                         v
real.csv               profile.json  +  CLI args -> simulated.csv        comparison report
```

## Install

```
python -m venv .venv
.venv\Scripts\activate   # or: source .venv/bin/activate
pip install -r requirements.txt
```

Everything is standard library except `numpy`, which is used narrowly for
numeric-column arrays and the Gaussian copula (see plan Sec 13).

Some deployments don't need this step at all (e.g. a locked-down environment
where `numpy` is already provided globally) - see
[docs/environment-notes.md](docs/environment-notes.md) for per-environment
specifics.

## Stage 0: export from SAS

Run `simulset/sas/export_to_csv.sas` in SAS 9.4 to produce `real.csv` from
your `.sas7bdat` dataset.

`PROC EXPORT` writes the CSV in your SAS session's encoding, not utf-8 -
check `%put &sysencoding;` if unsure. Both `profile_dataset.py` (`--input`)
and `generate_dataset.py` (`--real-csv`) default to reading `utf-8-sig`, but
accept `--input-encoding` to override it (e.g. `--input-encoding cp1252`,
common on Windows SAS installs).

## Stage 1: profile

```
python -m simulset.profile_dataset --input real.csv --output profile.json --log-level INFO
```

Column types, in the order they are checked:

- A column where every value is missing (blank, `NA`, `N/A`, `null`, `None`, `NaN`, `.`,
  any case, trimmed) is `empty`: the profile records only `dtype` and `missing_rate` 1.0,
  generation writes it as all blanks (even if named in `--use-real-values`), and
  validate and the leak check handle it (nothing to match). A WARNING names it.
- **All-digit ID/code columns** (at least 98% of non-missing values,
  trimmed, are only digits) are never numeric when at least one of these holds: some value has a
  leading zero; every value has the same length and it is 5 or more; distinct
  values are 95% or more of the non-missing values (this last test only applies
  with at least 50 non-missing values). At least 98% of values must be all
  digits, so a few dirty values (`UNKNOWN`, `-1`, ZIP+4) do not switch the rule
  off. If distinct/non-missing is
  0.95 or more the column is `identifier` (pattern-generated, never equal to a
  real value). Otherwise it is `categorical` (ZIP-like codes): output uses
  masked labels `{col}_valN` in the real proportions, however many distinct
  values there are. Values stay strings so leading zeros survive. Each such
  column logs a WARNING with its new type and the reason. Decimals, negatives
  and variable-length short numbers with plenty of repeats (counts, ages) stay numeric.
- Missing markers (the list above) count as missing in every column, not only
  for the `empty` type. `-nan`/`+nan` are markers too, and `inf`, `-inf`,
  `Infinity` and values like `1e999` are never read as numbers. They count as
  unparseable, not missing: under 2% of a column, those rows are generated as
  ordinary numbers; over 2%, the column stops being numeric (for example it
  becomes `free_text`).
- Then numeric, date, categorical, identifier, free_text as before.

Whole-dollar amounts that are almost all distinct (95% or more) also become
`identifier`: that is the safe default, because the profile cannot tell them from
IDs. If the column is really a number, keep it numeric with
`--treat-as-number col_a,col_b`. Repeating amounts stay numeric.

Produces `profile.json`: a portable, inspectable summary of the real data
(dtype, missing rate, min/max/mean/median/variance, percentiles, skew/
kurtosis, frequency tables, identifier patterns, and the numeric
cross-column correlation matrix). Review it before generating anything -
columns logged as `free_text`/`identifier` with a WARNING are the most
likely place to accidentally leak PII if resampled naively.

**Deviation from the plan's example schema (Sec 5.2):** in addition to the
six headline percentiles shown in the doc, each numeric/date column also
gets an `empirical_quantiles` (or `empirical_quantiles_epoch_day`) array of
1001 evenly-spaced quantile points. Six percentiles are too coarse a grid to
sample a realistic distribution from at generation time; this keeps
`profile.json` self-contained per Sec 3's design goal ("no need to keep
re-reading the real dataset") while still supporting good empirical-CDF
fidelity for Stage 2.

## Stage 2: generate

```
python -m simulset.generate_dataset \
  --profile profile.json \
  --n-records 100000 \
  --use-real-values color,state_code \
  --real-csv real.csv \
  --preserve-correlations true \
  --seed 42 \
  --output simulated.csv \
  --log-file generate.log --log-level INFO
```

- Non-flagged numeric/date columns: empirical-CDF (inverse-transform)
  sampling against the profiled quantile grid.
- Non-flagged numeric columns also get correlation preservation via a
  Gaussian copula, unless `--preserve-correlations false` or fewer than two
  non-flagged numeric columns have a correlation entry.
- Non-flagged categorical columns: never write real labels. Each real value
  gets a deterministic synthetic replacement (`{col}_val1`, `{col}_val2`, ...
  ordered by descending real frequency), sampled in the real proportions.
  Use `--randomize-category-labels` to shuffle the label order (still
  seeded), and `--emit-category-mapping` to write a real->synthetic lookup
  to `<output>_category_mapping.csv` for your own QA - keep that file
  separate from the output, since the real label names can be sensitive.
- Identifier/free-text columns: never resampled. Generated from the
  profiled positional pattern (e.g. `^P[0-9]{6}$`), with uniqueness enforced
  when the real column was fully unique.
- With `--real-csv`, identifier and free_text generation rejects any candidate
  equal to a real value of that column (compared after trimming whitespace).
  The real values are read from `--real-csv` only; they are never written to
  `profile.json` or the logs (logs carry counts only). If a pattern space is
  filled by real values, the value gets an `x<n>` suffix rather than match a
  real one, with a WARNING. Without `--real-csv` there is one WARNING per
  identifier/free_text column: collisions with real values were not prevented,
  so run the leak check before using the output.
- `--use-real-values`: bootstrap-samples literal real values in their real
  proportions. Categorical flagged columns use the frequency table already
  in `profile.json`; every other dtype needs `--real-csv` to re-read raw
  values (profile.json doesn't retain full raw value lists for non-categorical
  columns).

## Stage 3: validate (optional but recommended)

```
python -m simulset.validate_dataset --profile profile.json --simulated simulated.csv --report report.json
```

Re-profiles `simulated.csv` and prints/logs a side-by-side diff against
`profile.json`, including a correlation-matrix diff - the main check that
the copula step actually preserved cross-column relationships.

### Leak check

Run this in the secure environment, where the real CSV is, to prove the output
holds no real value:

```
python -m simulset.validate_dataset --profile profile.json --simulated simulated.csv \
  --real-csv real.csv --allow-real-values state_code --leak-check-columns provider_code \
  --report report.json --log-file validate.log
```

For every column the profile classifies as identifier, free_text or categorical,
plus any named in `--leak-check-columns`, it counts simulated values equal to a
real value in that column. Matching is strict, on the safe side: it ignores case
and surrounding spaces, and for all-digit values also leading zeros, so
`12345678` matches a real `012345678`. A false match only means you generate
again.

- A column in both CSVs that the profile does not have is checked too.
- A simulated column that is not in `real.csv` cannot be checked, so it counts
  as a failing column ("not in real.csv, not checked") unless listed in
  `--allow-real-values`.
- `--leak-check-columns` is for ID or code columns that simulset left numeric
  (for example after `--treat-as-number`). Do not name measure columns such as
  amounts: they match real values by chance and the check would fail.
- `--allow-real-values` lists columns where a match is expected (for example ones
  you passed to `generate_dataset --use-real-values`).
- `--input-encoding` works as in the other stages.

Only counts are printed, logged and written to the `leak_check` section of
`report.json`: column, number checked, number matching, whether allowed. The last
line is `LEAK CHECK: PASS` or `LEAK CHECK: FAIL (n columns)`, and the exit code is
1 on a fail. A named column missing from the profile or the CSVs is an error.

Only output from a run with `--real-csv` is safe to paste anywhere. In that mode
the stats diff leaves out `min`, `max` and `median` (literal values) and the
profiler's DEBUG stats line stays out of the log. Without `--real-csv`, validate
still prints real min, max and median and logs real labels, as before.

## What the output counts as

The real input and everything simulset writes about it stay as protected as the
real data, for good: `real.csv`, `profile.json`, the category mapping file, the
validation report and the log files.

The output CSV is free to use (shared inside the team, committed, read by approved
AI tools, used as test fixtures) only after a release check, done by a person where
the real data lives:

1. No column went through `--use-real-values` ([Stage 2](#stage-2-generate))
   unless the data owner approved that column by name because its values are public.
   The same goes for any column named in `--allow-real-values`.
2. The [leak check](#leak-check) ends `LEAK CHECK: PASS`. Name any ID or code
   column simulset left numeric in `--leak-check-columns`.
3. Number and date values at or near the real min or max that could point to one
   person or case are capped or approved by the data owner.
4. The file travels alone, with none of the side files: no `profile.json`, mapping
   file, validation report or log.

Record the check (date, simulset commit, seed, who checked, columns approved by
name) next to wherever the file is used.

A file that fails is fixed by changing the input or settings and generating again,
never by hand-editing the output.

Until a file passes, treat it like the real data. Output from running code on a
released file is free to use too.

## Layout

```
simulset/
  log_config.py        # shared logging setup
  mathutils.py          # erf / inverse-normal-CDF / rank helpers (no scipy)
  profile_dataset.py     # Stage 1 CLI
  generate_dataset.py    # Stage 2 CLI
  validate_dataset.py    # Stage 3 CLI
  profiling/
    classify.py           # column type/cardinality classification
    stats.py                # stats + correlation matrix
    patterns.py              # positional pattern inference for identifiers
  generation/
    numeric.py               # empirical-CDF interpolation sampler
    categorical.py             # synthetic-label frequency-weighted sampler
    identifiers.py              # pattern-based + uniqueness-enforced generator
    dates.py                     # empirical-CDF sampling on epoch-day
    copula.py                     # numpy-backed Cholesky + Gaussian copula
    flagged.py                     # --use-real-values bootstrap sampling
    missingness.py                  # per-column missing-value injector
  sas/
    export_to_csv.sas                # Stage 0 PROC EXPORT script
  tests/
```

## Known limitations (deferred per the plan's phasing, Sec 10)

- Missingness is injected independently per column; joint/co-missingness
  patterns aren't modeled (Phase 2).
- An all-empty column is profiled as `empty` and generated as all blanks; it
  carries no type, so if a later extract fills it, profile again.
- A real label spelled like a missing marker (`NA`, `Na` for sodium, `NaN`,
  `None`) is read as missing.
- Date columns don't re-weight for weekday/month seasonality (Phase 2).
- No config-file input or per-column strategy overrides (Phase 3).
- Both profiling and generation currently load full columns into memory
  (numpy arrays for numeric columns, Python lists otherwise) rather than
  streaming in chunks; fine well past 500K rows on typical hardware, but see
  plan Sec 13 if you outgrow it.

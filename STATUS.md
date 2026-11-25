# Status

## Waiting on Lauren

- (none)

## In progress

- (none)

## Ready

- Optional, unasked: `profile.json`'s `n_non_null` leaves out unparseable values, but `missing_rate` doesn't count them as missing, so the two don't add up to `row_count`. A docs note if Lauren wants one.

## Done

- The numeric and date parse warnings in `classify.py` now say unparseable values are left out of the stats and not counted as missing, matching Lauren's 2026-11-23 decision.
- Lauren decided, 2026-11-23: infinity values stay unparseable, not missing, so a number column with over 2% inf does not stay numeric. No code change.
- README section "What the output counts as": the release check for output CSVs and what stays protected (docs PR).
- how-we-work now matches these fixes: PR #13 (the release rule and the all-digit step) and PR #14 (empty columns and the NaN markers), both merged.
- `NaN`, `-nan` and `+nan` (any case, trimmed) are missing markers, and infinity values (`inf`, `Infinity`, `1e999`) are never read as numbers, so all-`NaN` columns are `empty` and stray `NaN`s in number columns count as missing (PR #7).
- All-empty columns: new `empty` type, profiled with missing_rate 1.0, generated as blanks, handled by validate and the leak check (PR #6).
- Leak check in `validate_dataset`: `--real-csv`, `--allow-real-values`, `--leak-check-columns`, counts only, exit 1 on a match (PR #5). Matching ignores case, spaces and leading zeros on all-digit values; columns the profile lacks are checked, or fail if not in `real.csv`.
- All-digit ID/code columns no longer profile as numeric: identifier or categorical, `--treat-as-number`, and generation rejects real-value collisions with `--real-csv` (PR #4).
- Renamed `simdata/` to `simulset/`, cp1252 CSV input, SAS-restricted setup notes (PR #2).
- First version of the generator (PR #1).

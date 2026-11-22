# Status

## Waiting on Lauren

- (none)

## In progress

- (none)

## Ready

- **All-`NaN` column crashes profiling.** A column of all `NaN` (not a missing marker) passes the empty check and stops profiling with "cannot convert float NaN to integer" (also on main). Done when a test with an all-`NaN` column profiles and generates without error. Found by qa-peer, 2026-11-22.
- After the fixes are released, open a how-we-work PR to relax the step in its `docs/data-and-ai.md` that says to drop these columns before profiling.

## Done

- All-empty columns: new `empty` type, profiled with missing_rate 1.0, generated as blanks, handled by validate and the leak check (PR #6).
- Leak check in `validate_dataset`: `--real-csv`, `--allow-real-values`, `--leak-check-columns`, counts only, exit 1 on a match (PR #5). Matching ignores case, spaces and leading zeros on all-digit values; columns the profile lacks are checked, or fail if not in `real.csv`.
- All-digit ID/code columns no longer profile as numeric: identifier or categorical, `--treat-as-number`, and generation rejects real-value collisions with `--real-csv` (PR #4).
- Renamed `simdata/` to `simulset/`, cp1252 CSV input, SAS-restricted setup notes (PR #2).
- First version of the generator (PR #1).

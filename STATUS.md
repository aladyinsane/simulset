# Status

## Waiting on Lauren

- (none)

## In progress

- (none)

## Ready

- **All-empty column crashes profiling.** When every value is missing, `classify_column` logs "column is entirely missing/empty; defaulting to numeric/float" and returns numeric with no values, and profiling then stops with `IndexError`. Fix: handle the empty case (for example an `empty` type that generates all blanks) so the run finishes. Done when a test with an all-empty column profiles and generates without error. Found 2026-11-21.
- After the fixes are released, open a how-we-work PR to relax the step in its `docs/data-and-ai.md` that says to drop these columns before profiling.

## Done

- Leak check in `validate_dataset`: `--real-csv`, `--allow-real-values`, `--leak-check-columns`, counts only, exit 1 on a match (PR #5).
- All-digit ID/code columns no longer profile as numeric: identifier or categorical, `--treat-as-number`, and generation rejects real-value collisions with `--real-csv` (PR #4).
- Renamed `simdata/` to `simulset/`, cp1252 CSV input, SAS-restricted setup notes (PR #2).
- First version of the generator (PR #1).

# Status

## Waiting on Lauren

- (none)

## In progress

- (none)

## Ready

- **Leak check in `validate_dataset`.** `validate_dataset` gets `--real-csv` (and `--input-encoding`), `--allow-real-values col,...` and `--leak-check-columns col,...`. For every column the profile classifies as identifier, free_text or categorical, plus the named columns, it counts output values equal to a real value in that column (string compare after trim). It prints and reports only per-column counts, never values. It exits 1 if any checked column not in `--allow-real-values` has a count above 0. Done when a planted real value fails the check, a clean file passes, and no real value appears in stdout, log or report.
- **All-empty column crashes profiling.** When every value is missing, `classify_column` logs "column is entirely missing/empty; defaulting to numeric/float" and returns numeric with no values, and profiling then stops with `IndexError`. Fix: handle the empty case (for example an `empty` type that generates all blanks) so the run finishes. Done when a test with an all-empty column profiles and generates without error. Found 2026-11-21.
- After the fixes are released, open a how-we-work PR to relax the step in its `docs/data-and-ai.md` that says to drop these columns before profiling.

## Done

- All-digit ID/code columns no longer profile as numeric: identifier or categorical, `--treat-as-number`, and generation rejects real-value collisions with `--real-csv` (PR #4).
- Renamed `simdata/` to `simulset/`, cp1252 CSV input, SAS-restricted setup notes (PR #2).
- First version of the generator (PR #1).

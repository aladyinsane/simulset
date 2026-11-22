# Status

## Waiting on Lauren

- (none)

## In progress

- (none)

## Ready

- **All-digit ID columns leak real values.** `classify_column` (`simulset/profiling/classify.py`) checks for numeric before anything else, so a column of all-digit IDs (member ID, NPI, SSN, ZIP, phone) becomes numeric/integer with no warning. Generation then draws from its quantiles, and synthetic values can equal real IDs: a re-implementation of that step matched 0 to 577 real IDs per 10,000 rows. The plan (`docs/simulated_dataset_generator_plan.md`, the pattern row in section 5.1) meant ZIP-like columns to keep their pattern instead. Fix: before the numeric check, treat integer-formatted columns that are fully unique or near it, fixed length, or have leading zeros as `identifier` (pattern-generated), and log a warning. Done when a test with an all-digit ID column profiles it as `identifier` and no synthetic value equals a real one. Found while building how-we-work, 2026-11-21.
- **All-empty column crashes profiling.** When every value is missing, `classify_column` logs "column is entirely missing/empty; defaulting to numeric/float" and returns numeric with no values, and profiling then stops with `IndexError`. Fix: handle the empty case (for example an `empty` type that generates all blanks) so the run finishes. Done when a test with an all-empty column profiles and generates without error. Found 2026-11-21.
- After either fix is released, open a how-we-work PR to relax the step in its `docs/data-and-ai.md` that says to drop these columns before profiling.

## Done

- Renamed `simdata/` to `simulset/`, cp1252 CSV input, SAS-restricted setup notes (PR #2).
- First version of the generator (PR #1).

# Environment notes

Per-deployment quirks that don't belong in the general README because they
don't hold everywhere `simulset` runs. Add a new section per environment as
they come up.

## Restricted SAS environment (no external internet access)

- **`pip install` is blocked** - this environment has no external internet
  access, so `requirements.txt` and the README's `pip install -r
  requirements.txt` step don't apply here. Leave both as-is for other
  environments (a normal dev machine, CI, etc.) where `pip install` works
  fine.
- **`numpy` is already available**, installed in the global/site Python that
  ships with this SAS environment, so `simulset` works here without any
  install step - just run the modules directly (`python -m
  simulset.profile_dataset ...`) using that Python.
- **CSV encoding**: this environment's SAS session exports `real.csv` as
  `cp1252`, not the tool's default `utf-8-sig`. Pass `--input-encoding
  cp1252` to `profile_dataset.py` and to `generate_dataset.py` (when using
  `--real-csv`). See the README's Stage 0 section for the general
  explanation of why this varies by SAS session.

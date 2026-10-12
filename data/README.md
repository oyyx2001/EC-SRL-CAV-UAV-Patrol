# Withheld task-registry inputs

The geometry-derived task registries are **not included in the public release** pending verification of the source map's redistribution terms. They are not MIT-licensed.

The unchanged configs expect:

- `data/xinjiekou/edge_task_registry.csv`
- `data/baijiahu/edge_task_registry.csv`

The author's local copies remain on disk but are untracked and ignored. Obtain lawful access to the matching files before evaluation. A replacement registry or synthetic scene does not reproduce the reported original-case results.

Run `python -m tools.check_data_assets` after restoring authorized inputs. It checks the reference input hashes without downloading data or changing simulation behavior. See `DATA_AVAILABILITY.md` and `DATA_PROVENANCE.md`.

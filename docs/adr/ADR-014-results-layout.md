# ADR-014: Results directory layout

**Status:** Accepted (2026-05-01)
**Supersedes:** none.
**Owners:** training subsystem (`PRD_training_evaluation.md § 6.1`), SDK (`PLAN.md § 6.1`).

## Context

Each training run produces four output files: a best-checkpoint weights file, a final-checkpoint weights file, an epoch log, and a JSON results record. These files must be organized in a way that:

- Supports multiple back-to-back runs (e.g., three model kinds × multiple seeds in the experiment grid).
- Makes the provenance of each run clear from the directory name alone.
- Sorts naturally on the filesystem so that `ls results/` gives a useful order.
- Prevents silent overwriting of prior runs.

## Decision

```text
results/<utc_timestamp>__<model_kind>__<seed>/
    checkpoint_best.pt      # best-val-MSE state dict + epoch metadata
    checkpoint_final.pt     # final-epoch state dict + epoch metadata
    train.log               # tab-separated epoch table, one row per epoch
    results.json            # full result record (schema: PRD_training_evaluation.md § 10.4)
```

- `<utc_timestamp>` format: `%Y%m%dT%H%M%SZ` (e.g. `20260501T143022Z`). ISO-8601-like; no colons or spaces; lexicographic sort = chronological sort.
- `<model_kind>` ∈ `{"fc", "rnn", "lstm"}`.
- `<seed>` is the integer runtime seed used for the run.
- The double-underscore `__` delimiter makes the three fields unambiguously parseable by `str.split("__")` without regex — colons and hyphens would collide with other timestamp characters.

The SDK creates `run_dir` with `mkdir(exist_ok=False)` before calling `services.training.train()`. A pre-existing directory raises `FileExistsError`, preventing silent overwrite.

## Alternatives Considered

**(a) Per-experiment-name directories (e.g., `results/fc_run1/`).**
*Rejected.* Names are not unique on re-runs. Either overwrite (data loss) or a collision-avoidance scheme (suffix counter) — both are worse than a timestamp.

**(b) Flat `results/` directory with per-file naming (e.g., `results/20260501T143022Z__fc__1337_checkpoint_best.pt`).**
*Rejected.* Four files per run in a flat directory becomes unnavigable after 10+ runs (40+ files). Subdirectories are the natural unit of a run.

**(c) Shared log file across runs (`results/all_runs.log`).**
*Rejected.* Filesystem contention when runs overlap (even sequentially, it is a shared mutable resource). Per-run `train.log` files make it easy to open exactly the log for a specific run without filtering.

**(d) Content-addressed naming (hash of config).**
*Rejected.* A hash does not sort chronologically; two runs with the same config (different seeds) would collide unless the seed is in the hash input; debugging is harder because the name is opaque.

## Consequences

**Positive:**
- `ls results/` is sorted chronologically. The most recent run is last (or first with `ls -r`).
- `results/<timestamp>__rnn__1337/results.json` is self-documenting: model kind and seed are in the path.
- The `T-TR-10` unit test (`PRD_training_evaluation.md § 12.1`) verifies the name matches the regex `r"\d{8}T\d{6}Z__[a-z]+__\d+"` — machine-checkable from the test suite.

**Negative:**
- Two runs started within the same UTC second would collide. Mitigation: sequential runs in the grid runner (no parallelism in v1.00) guarantee distinct timestamps; the `exist_ok=False` guard catches the degenerate case.

## References

- `docs/PRD_training_evaluation.md` v1.01 § 6.1 (run directory), § 12.1 T-TR-10 (format test).
- `docs/PLAN.md` v1.02 § 14 (resolution 4: results layout), § 13.1 (ADR index).

# EXP-000 — Pipeline smoke

**Date:** 2026-05-02
**Owner:** auto-run before EXP-001 launch (ADR-007 gate).
**Purpose:** Verify the SDK→train→evaluate pipeline runs cleanly on a tiny corpus for all three model kinds before committing compute to the full EXP-001 grid. Not a research experiment — just a green-light check.

---

## Configuration

Default `config/setup.json` with the following overrides applied via `ExperimentSpec.overrides`:

| Key | Override |
| --- | --- |
| `dataset.n_train` | 1 000 |
| `dataset.n_val` | 200 |
| `dataset.n_test` | 200 |
| `training.epochs` | 1 |
| `training.early_stop_patience` | 0 (disabled — only 1 epoch anyway) |

Seed: `1337` (config default). Device: `cpu`. Run-dir root: `results/EXP-000-smoke/`.

## Pass criteria

- All three runs complete without exceptions.
- `overall_test_mse` is finite for each.
- Each run completes in under 30 s wall-clock (the smoke gate threshold).
- Each run-dir contains `checkpoint_best.pt`, `checkpoint_final.pt`, `train.log`, `results.json`, `result.pkl`.

## Results

| Kind | Wall-clock (s) | Per-epoch (s) | overall_test_mse | best_val_mse | finite |
| --- | ---: | ---: | ---: | ---: | :---: |
| fc   | 9.08 | 0.082 | 0.4880 | 0.4920 | ✓ |
| rnn  | 0.09 | 0.027 | 0.4817 | 0.4903 | ✓ |
| lstm | 0.14 | 0.077 | 0.4836 | 0.4869 | ✓ |

Notes:
- The 9-second wall-clock on FC is one-time PyTorch import / module-cache initialization paid by the first SDK construction in the process; subsequent SDK constructions see ~0.1 s. Per-epoch elapsed (from `train.log`) is the load-bearing number and is uniform at 0.027–0.082 s across all three kinds.
- Overall test MSE near 0.49 across all kinds is consistent with 1 epoch of Adam on 1 000 noisy training examples with phase noise β = 2π — the models have not had a chance to learn anything beyond the bias term.
- All four expected files plus `result.pkl` (added per the user's analysis-ergonomics request) are present in each run-dir.

## Verdict

**SMOKE STATUS: GREEN.** EXP-001 cleared to launch.

Projected EXP-001 wall-clock from this smoke: each full run is ~30 epochs × per-epoch (~0.1 s on the smoke corpus, will scale roughly linearly to 30 000 train + 3 750 val examples ⇒ ~3 s/epoch ⇒ ~90 s/run). Total for 9 runs (3 kinds × 3 seeds): under 15 minutes. Well within the agreed 3-hour ceiling.

## Artefacts

- `results/EXP-000-smoke/smoke_log.json` — machine-readable record.
- `results/EXP-000-smoke/<run_dir>/` per run.

# EXP-002 — Phase-noise (β) sweep

**Date drafted:** 2026-05-02 (planned, before run).
**Owner:** auto-run after EXP-001's floor finding (see "Motivation" below).
**Predecessor:** `EXP-001-baseline-3seeds.md` (locked-default-config grid).
**Successor:** `EXP-003-baseline-3seeds-beta-<chosen>.md` (re-run of the three-model comparison at the β chosen by this sweep).

---

## Motivation

EXP-001 ran the three-way baseline at the PRD-locked defaults (α=0.05, β=2π, Gaussian per-sample). All three architectures converged to overall test MSE ≈ 0.50 — the variance of a unit-amplitude sinusoid, i.e. the **near-zero-predictor floor**. This is not a thesis failure but a task-difficulty failure: at β=2π per-sample, the instantaneous phase argument has standard deviation 2π rad, fully scrambling within-window structure faster than any of FC / RNN / LSTM can recover at lr=1e-3 in 30 epochs.

The lecturer's RNN-vs-LSTM thesis presupposes the models can extract *something* from the noisy mixture. EXP-001 evidence: not at the locked default. The comparative study is informative only at a β where the task is **learnable but non-trivial** — far enough above β=0 that there is real noise to denoise, but far enough below β=2π that the signal hasn't been destroyed.

This experiment maps the difficulty curve.

## Hypothesis

- At **β=0**: task reduces to amplitude-only denoising. All three models should learn it; FC alone is sufficient. Overall test MSE should drop well below 0.30.
- At **β=2π**: task is unsolvable for our models in 30 epochs (EXP-001 evidence). Overall test MSE ≈ 0.50.
- Between those endpoints, MSE rises monotonically with β. Somewhere on the curve there is a **β_max_useful** — the largest β for which the task is meaningfully learnable.

## Sweep

| β index | β (string in config) | β (numeric) |
| ---: | --- | ---: |
| 0 | `"0"` | 0 |
| 1 | `"pi/8"` | 0.3927 |
| 2 | `"pi/4"` | 0.7854 |
| 3 | `"pi/2"` | 1.5708 |
| 4 | `"pi"` | 3.1416 |
| 5 | `"2*pi"` | 6.2832 |

**Probe model:** FC (the cheapest of the three; trainability is what we are mapping, not architecture sensitivity).
**Seed:** 1337 (fixed across all six runs — we want a difficulty curve, not an uncertainty band).
**All other config:** defaults (`config/setup.json`). 30 epochs, batch 256, Adam lr=1e-3, patience 5.
**Override surface:** `ExperimentSpec.overrides = {"signal.noise.beta": <β string>}`. PRD_training_evaluation.md § 9.4.

**Total runs:** 6.

## Acceptance gate

Identify the **largest β value** for which `overall_test_mse ≤ 0.30` — call this `β_max_useful`. The 0.30 threshold corresponds to recovering ≥ 40 % of the signal variance (`(0.50 − 0.30) / 0.50 = 0.40`). Below 0.30 we are confident the task is learnable; above 0.30 we are in the floor regime.

If no β crosses 0.30, the sweep is reported as inconclusive and we discuss in the next collaborator session before launching EXP-003.

## Cost budget

EXP-001's 9 runs at default config took 88.7 s wall-clock (~10 s per run). Six FC runs at the same scale: ~60 s. **Hard ceiling: 6 minutes.** If any single run exceeds 30 minutes it is aborted (same gate as ADR-007).

## What this sweep does NOT do

- It does not change the locked default in `config/setup.json` (β stays "2*pi" — ADR-005 is unchanged). The chosen `β_max_useful` is applied to EXP-003 via the override mechanism only.
- It does not tune any other hyperparameter (lr, batch size, epochs, patience). Only β changes.
- It does not test the thesis. Architectural comparison comes in EXP-003 at the chosen β.

## Results

Run 2026-05-02. 6 runs in **62.2 s** wall-clock. All finite. Per-run wall-clock 5–19 s (cold-start torch-init paid once).

| β index | β | β (numeric) | overall test MSE | best_epoch / epochs_run |
| ---: | --- | ---: | ---: | ---: |
| 0 | `0`     | 0.0000 | **0.1184** | 29 / 30 |
| 1 | `pi/8`  | 0.3927 | **0.1654** | 23 / 29 |
| 2 | `pi/4`  | 0.7854 | **0.2620** | 28 / 30 |
| 3 | `pi/2`  | 1.5708 | **0.4299** | 26 / 30 |
| 4 | `pi`    | 3.1416 | **0.4938** |  8 / 14 |
| 5 | `2*pi`  | 6.2832 | **0.4963** | 15 / 21 |

**Curve shape.** Strictly monotone in β: every increase in phase noise raises MSE. The transition from "learnable" to "floor" happens between β=π/4 and β=π/2 — MSE jumps from 0.2620 to 0.4299, which is most of the way to the 0.50 zero-predictor floor. By β=π the model has effectively given up (best epoch 8 of 14, early-stop fires; train.log shows no further improvement). EXP-001's β=2π result (MSE 0.4963 at the same FC seed) reproduces here within 0.0001, confirming the EXP-001 baseline is part of the same curve.

**Hypothesis confirmed.** The β=0 / β=2π endpoints behave as predicted; the comparative regime is at intermediate β.

## β_max_useful

**β_max_useful = `pi/4` (numeric ≈ 0.7854).** This is the **largest** β value at which FC achieves overall test MSE ≤ 0.30 — specifically 0.2620, recovering ~48 % of the signal variance relative to the zero-predictor floor.

The next β value up (`pi/2` → MSE 0.4299) is past the gate. Choosing `pi/4` therefore puts EXP-003 in the regime where the task is **non-trivial but learnable** — exactly the regime in which the architectural comparison can be informative.

The choice is locked here, in writing, before EXP-003 runs. There is no retroactive tuning: if EXP-003's results disappoint, we do not slide β downward to make them better.

## Artefacts

- `results/EXP-002-beta-sweep/<beta_index>_<beta_str>/<run_dir>/` per β.
- `results/EXP-002-beta-sweep/sweep_log.json` — machine-readable record.

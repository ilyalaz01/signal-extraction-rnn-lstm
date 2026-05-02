# EXP-003 — Baseline three-way comparison at β=π/4 (3 seeds)

**Date:** 2026-05-02.
**Owner:** auto-run after EXP-002 selected `β_max_useful = pi/4`.
**Predecessor:** `EXP-002-beta-sweep.md` (chose β=π/4 as the largest β with FC overall MSE ≤ 0.30).
**Status:** Numerical results captured. **Outcome verdict deliberately deferred — see "Verdict" section.**

---

## Configuration

- `config/setup.json` defaults, with **one override**: `signal.noise.beta = "pi/4"`.
- Everything else as in EXP-001: 30 000 / 3 750 / 3 750 train/val/test, window 10, α=0.05, Gaussian per-sample.
- Adam lr=1e-3, batch 256, 30 epochs max, early-stop patience 5.
- Three architectures × three seeds (ADR-007): `kinds = (fc, rnn, lstm)`; `seeds = (1337, 1338, 1339)`. Total: **9 runs**.
- Device: CPU. Wall-clock total: **109.2 s** (1.8 min) — within the 30 min/run, 3 h/grid ceilings.

The β=2π default in `config/setup.json` and ADR-005 are **unchanged**; EXP-001 stands as the β=2π reference. EXP-003 applies the override pattern PRD_training_evaluation.md § 9.4 was designed for.

## Raw results

| kind | seed | elapsed (s) | best_epoch / epochs_run | best_val_mse | overall_test_mse |
| --- | --- | ---: | ---: | ---: | ---: |
| fc   | 1337 | 18.5 | 28 / 30 | 0.2585 | 0.2620 |
| fc   | 1338 |  7.9 | 18 / 24 | 0.2643 | 0.2740 |
| fc   | 1339 |  8.5 | 19 / 25 | 0.2604 | 0.2661 |
| rnn  | 1337 |  8.3 | 10 / 16 | 0.2649 | 0.2700 |
| rnn  | 1338 |  9.9 | 14 / 20 | 0.2734 | 0.2814 |
| rnn  | 1339 |  9.8 | 14 / 20 | 0.2680 | 0.2743 |
| lstm | 1337 | 18.6 | 27 / 30 | 0.2611 | 0.2630 |
| lstm | 1338 | 19.0 | 25 / 30 | 0.2734 | 0.2776 |
| lstm | 1339 |  8.7 |  7 / 13 | 0.2685 | 0.2720 |

(See `results/EXP-003-baseline-3seeds-beta-pi-4/grid_log.json` and per-run `result.pkl`.)

## Per-cell MSE (mean ± std across 3 seeds)

| kind | 2 Hz | 10 Hz | 50 Hz | 200 Hz |
| --- | --- | --- | --- | --- |
| fc   | 0.3105 ± 0.0036 | 0.3153 ± 0.0251 | 0.2700 ± 0.0157 | 0.1718 ± 0.0063 |
| rnn  | 0.3191 ± 0.0087 | 0.3209 ± 0.0228 | 0.2817 ± 0.0164 | 0.1774 ± 0.0064 |
| lstm | 0.3143 ± 0.0081 | 0.3183 ± 0.0240 | 0.2761 ± 0.0141 | 0.1730 ± 0.0083 |

**Observed seed-std summary:**
- Maximum per-cell std `0.0251` (fc @ 10 Hz) ≈ `~8 %` of MSE scale at that cell.
- Most cells: std ≈ 0.005–0.015 (1.5–6 % of MSE). Higher noise per cell than EXP-001 (where the floor compressed the variance).
- The 10 % threshold rule of thumb (PRD § 8.3, "≥ 2× seed-std") is **right at the edge** for the worst cell — within reach but no longer comfortable. Worth your call on whether to revise.

## Overall test MSE per kind

| kind | seed=1337 | seed=1338 | seed=1339 | mean | std |
| --- | ---: | ---: | ---: | ---: | ---: |
| fc   | 0.2620 | 0.2740 | 0.2661 | **0.2674** | 0.0061 |
| rnn  | 0.2700 | 0.2814 | 0.2743 | **0.2752** | 0.0057 |
| lstm | 0.2630 | 0.2776 | 0.2720 | **0.2709** | 0.0074 |

Ordering on the overall mean: **FC < LSTM < RNN**. (Difference between FC and LSTM is ~0.5 σ; between LSTM and RNN is ~0.6 σ.)

## LSTM vs RNN: relative improvement per frequency (PRD § 8.2)

`rel(k) = (MSE_RNN,k − MSE_LSTM,k) / MSE_RNN,k × 100` — positive ⇒ LSTM better.

| Sinusoid | Freq (Hz) | RNN MSE | LSTM MSE | rel(k) | ≥ 10 % threshold? |
| --- | ---: | ---: | ---: | ---: | :---: |
| s_1 | 2   | 0.3191 | 0.3143 | **+1.50 %** | no |
| s_2 | 10  | 0.3209 | 0.3183 | **+0.81 %** | no |
| s_3 | 50  | 0.2817 | 0.2761 | **+2.00 %** | no |
| s_4 | 200 | 0.1774 | 0.1730 | **+2.48 %** | no |

**Sign of `rel(k)` is positive at every frequency** (LSTM ≤ RNN at every cell), but no cell crosses the 10 % threshold. The magnitude is *larger* at higher frequencies — opposite of the thesis prediction.

## RNN vs FC at 200 Hz (Outcome A check)

`rel_FC_RNN(s_4) = (MSE_RNN,3 − MSE_FC,3) / MSE_FC,3 × 100`

- `fc @ 200 Hz = 0.1718`
- `rnn @ 200 Hz = 0.1774`
- `rel_FC_RNN(s_4) = +3.28 %` (RNN slightly worse than FC at 200 Hz; |rel| < 10 %.)

## Spearman ρ (PRD § 8.6)

Rank correlation between `FREQUENCIES_HZ = [2, 10, 50, 200]` and `rel(k)`. Negative ρ ⇒ thesis direction.

- `rel(k)` series ranked low→high: `s_2 (0.81) < s_1 (1.50) < s_3 (2.00) < s_4 (2.48)`.
- **`ρ = +0.8000`** (note: positive — opposite of the thesis direction).
- One-sided p-value (H₁: ρ < 0) via exact permutation, N=4: `p = 0.9583`. The thesis-direction monotonicity test **fails strongly** (the observed correlation is in the upper-tail end of the null distribution; only one of 24 permutations is more positive).

## Cross-frequency observation (no verdict)

At β=π/4 all three models do **substantially better at 200 Hz than at 2 Hz** (per-cell mean MSE 0.17 vs 0.31). One plausible reading is that high-frequency content is the easiest to band-isolate from the four-component mixture (uniquely-frequency-coded signal), while low-frequency targets must be disentangled from neighbouring slow components. But this is the kind of interpretation that is for the collaborator session, not for this doc.

## Comparison with EXP-001 (β=2π)

| Metric | EXP-001 (β=2π) | EXP-003 (β=π/4) |
| --- | --- | --- |
| FC overall test MSE (mean) | 0.4993 ± 0.0050 | **0.2674** ± 0.0061 |
| RNN overall test MSE (mean) | 0.4975 ± 0.0025 | **0.2752** ± 0.0057 |
| LSTM overall test MSE (mean) | 0.4957 ± 0.0043 | **0.2709** ± 0.0074 |
| Worst-cell rel(k) magnitude | 0.63 % (s_3) | 2.48 % (s_4) |
| Spearman ρ on rel(k) | −0.40 (p=0.375) | +0.80 (p=0.958) |
| Wall-clock | 88.7 s | 109.2 s |

EXP-003 is in a regime where models do real work; EXP-001 was at the floor.

## Verdict

**Deferred. Returning to collaborator** for the Outcome A/B/C/D classification, the README "Thesis evaluation" template-sentence draft, the call on whether the 10 % threshold needs revision (worst-cell std now ~8 % of MSE), and the framing for the README chapter that uses *both* EXP-002 (β-regime curve) and EXP-003 (architectural comparison) as data points.

## Artefacts

- `results/EXP-003-baseline-3seeds-beta-pi-4/grid_log.json` — per-run record (9 entries).
- `results/EXP-003-baseline-3seeds-beta-pi-4/summary.json` — derived per-cell mean/std and rel/ρ block.
- `results/EXP-003-baseline-3seeds-beta-pi-4/<run_dir>/` per run — `checkpoint_best.pt`, `checkpoint_final.pt`, `train.log`, `results.json`, `result.pkl`.

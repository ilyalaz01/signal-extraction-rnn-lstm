# EXP-001 — Baseline three-way comparison (FC / RNN / LSTM, 3 seeds)

**Date:** 2026-05-02
**Owner:** auto-run after EXP-000 smoke (ADR-007 grid).
**Status:** Numerical results captured. **Outcome verdict deliberately deferred — see "Verdict" section.**

---

## Configuration

- `config/setup.json` defaults, no overrides.
- 30 000 train / 3 750 val / 3 750 test windows; window length 10; full noise (α=0.05, β=2π, Gaussian per-sample).
- Adam lr=1e-3, batch size 256, 30 epochs max, early-stopping patience 5.
- Three architectures × three seeds (per ADR-007): `kinds = (fc, rnn, lstm)`; `seeds = (1337, 1338, 1339)`. Total: **9 runs**.
- Device: CPU. Wall-clock total: **88.7 s** (1.5 min) — well under the 30 min/run and 3 h/grid ceilings.

## Raw results

All nine runs, sorted by (kind, seed). `best_epoch / epochs_run` shows where the best val MSE was reached and how many epochs ran before early stopping fired (or 30 if it didn't).

| kind | seed | elapsed (s) | best_epoch / epochs_run | best_val_mse | overall_test_mse |
| --- | --- | ---: | ---: | ---: | ---: |
| fc   | 1337 | 15.3 | 15 / 21 | 0.4865 | 0.4963 |
| fc   | 1338 |  4.3 |  7 / 13 | 0.4885 | 0.4965 |
| fc   | 1339 |  9.7 | 25 / 30 | 0.4928 | 0.5050 |
| rnn  | 1337 |  4.2 |  2 /  8 | 0.4882 | 0.4954 |
| rnn  | 1338 |  4.9 |  3 /  9 | 0.4895 | 0.4967 |
| rnn  | 1339 |  6.7 |  7 / 13 | 0.4961 | 0.5003 |
| lstm | 1337 | 19.2 | 28 / 30 | 0.4814 | 0.4913 |
| lstm | 1338 | 10.8 | 10 / 16 | 0.4925 | 0.4959 |
| lstm | 1339 | 13.4 | 15 / 21 | 0.4992 | 0.4999 |

(See `results/EXP-001-baseline-3seeds/grid_log.json` and per-run `result.pkl` files for the canonical numbers.)

## Per-cell MSE (mean ± std across 3 seeds)

| kind | 2 Hz | 10 Hz | 50 Hz | 200 Hz |
| --- | --- | --- | --- | --- |
| fc   | 0.4858 ± 0.0143 | 0.5039 ± 0.0052 | 0.5013 ± 0.0014 | 0.5059 ± 0.0061 |
| rnn  | 0.4891 ± 0.0099 | 0.4996 ± 0.0022 | 0.5003 ± 0.0008 | 0.5008 ± 0.0005 |
| lstm | 0.4867 ± 0.0135 | 0.4987 ± 0.0033 | 0.4972 ± 0.0026 | 0.5002 ± 0.0000 |

**Observed seed-std summary** (relevant to PRD § 8.3 threshold re-validation):
- Maximum per-cell std is `0.0143` (fc @ 2 Hz). Relative to the `~0.50` MSE scale that is `~2.9%`.
- Maximum *relative* per-cell std is `~3.0%` (fc @ 2 Hz, std/mean ratio).
- The 10 % threshold in § 8.3 is therefore `~3×` the observed seed-std at the most-variable cell — **above** the `2× seed-std` rule of thumb. The threshold does not require revision based on this evidence (subject to your call).

## Overall test MSE per kind

| kind | seed=1337 | seed=1338 | seed=1339 | mean | std |
| --- | ---: | ---: | ---: | ---: | ---: |
| fc   | 0.4963 | 0.4965 | 0.5050 | 0.4993 | 0.0050 |
| rnn  | 0.4954 | 0.4967 | 0.5003 | 0.4975 | 0.0025 |
| lstm | 0.4913 | 0.4959 | 0.4999 | 0.4957 | 0.0043 |

LSTM ≤ RNN ≤ FC on the overall MSE (means), but differences are within ~1 σ of each other.

## LSTM vs RNN: relative improvement per frequency (PRD § 8.2)

`rel(k) = (MSE_RNN,k − MSE_LSTM,k) / MSE_RNN,k × 100` — positive ⇒ LSTM better.

| Sinusoid | Freq (Hz) | RNN MSE | LSTM MSE | rel(k) | ≥ 10 % threshold? |
| --- | ---: | ---: | ---: | ---: | :---: |
| s_1 | 2   | 0.4891 | 0.4867 | **+0.50 %** | no |
| s_2 | 10  | 0.4996 | 0.4987 | **+0.17 %** | no |
| s_3 | 50  | 0.5003 | 0.4972 | **+0.63 %** | no |
| s_4 | 200 | 0.5008 | 0.5002 | **+0.13 %** | no |

## RNN vs FC at 200 Hz (Outcome A check)

`rel_FC_RNN(s_4) = (MSE_RNN,3 − MSE_FC,3) / MSE_FC,3 × 100`

- `fc @ 200 Hz = 0.5059`
- `rnn @ 200 Hz = 0.5008`
- `rel_FC_RNN(s_4) = −1.01 %` (RNN very slightly better than FC; |rel| < 10 %.)

## Spearman ρ (PRD § 8.6)

Rank correlation between `FREQUENCIES_HZ = [2, 10, 50, 200]` and the LSTM-vs-RNN `rel(k)` series. Negative ρ ⇒ LSTM advantage *decreases* with frequency (the thesis direction).

- `rel(k)` series (ranked low→high): `s_4 (0.13) < s_2 (0.17) < s_1 (0.50) < s_3 (0.63)`.
- `ρ = −0.4000` (computed with the standard six-d² formula at N=4).
- One-sided p-value via exact permutation (24 permutations of N=4): `p = 0.3750`.

The PRD threshold for Outcome A's monotonicity sub-test is `p < 0.10`. With N=4 exact, `p < 0.10` requires `ρ = −1` (perfect inverse ordering) — the test is intentionally strict at this sample size.

## Verdict

**Deferred. Returning to collaborator** for thesis evaluation per ADR-007 / PRD_training_evaluation.md § 8.4. The Outcome A/B/C/D classification, the README "Thesis evaluation" template-sentence draft, and the call on whether to revise the 10 % threshold or launch ablations are deliberately not made here.

## Artefacts

- `results/EXP-001-baseline-3seeds/grid_log.json` — per-run record (9 entries).
- `results/EXP-001-baseline-3seeds/summary.json` — derived per-cell mean/std and the rel/ρ block above.
- `results/EXP-001-baseline-3seeds/<run_dir>/` per run — `checkpoint_best.pt`, `checkpoint_final.pt`, `train.log`, `results.json`, `result.pkl`.

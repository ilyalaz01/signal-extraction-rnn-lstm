# EXP-008 — Parameter-matched RNN at β=π/4 (3 seeds)

**Date drafted:** 2026-05-03 (planned, before run).
**Owner:** auto-run after EXP-003 surfaced a small-but-consistent LSTM ≥ RNN gap.
**Predecessor:** `EXP-003-baseline-3seeds-beta-pi-4.md` (architectural comparison at β=π/4, equal hidden=64).
**Companion:** `PRD_models.md` v1.01 § 5.3 / § 13 — defines this ablation as `EXP-008`.

---

## Motivation

EXP-003 at β=π/4 produced LSTM-vs-RNN rel(k) values of +0.81 % to +2.48 % across the four frequencies — LSTM is cell-by-cell better at every frequency, in the sign direction the lecturer's thesis predicts, but with magnitude small and the trend *opposite* to the predicted regime split (LSTM advantage grows with frequency rather than concentrating at low frequency).

A confound: at hidden=64, LSTM has ~3.6× the parameters of the vanilla RNN (PRD § 5.3 v1.01 records 18,826 vs 5,194). The PRD explicitly calls out this asymmetry as `EXP-008` — a parameter-matched comparison where the RNN's hidden size is enlarged so that its parameter count is close to the LSTM's.

If the LSTM-vs-RNN gap **closes** when the RNN has comparable capacity, the gap was capacity, not gating. If it **persists**, gating is doing real work even after capacity is matched.

## Hypothesis

H0: at hidden_RNN = 128 (capacity-matched), LSTM-vs-RNN-128 rel(k) at β=π/4 will be within ±2 σ of zero at every frequency cell — consistent with "the EXP-003 gap was capacity."

H1: at hidden_RNN = 128, LSTM still beats RNN-128 cell-by-cell with sign and magnitude similar to EXP-003 — consistent with "gating contributes."

The collaborator session decides between these readings; EXP-008 produces the numbers that distinguish them.

## Parameter-count math

Vanilla RNN at hidden_size = h, input_size = 5, seq-to-vector head:

```
n_params(RNN, h) = nn.RNN(5, h, 1) + Linear(h, 10)
                 = (5h + h^2 + 2h) + (10h + 10)
                 = h^2 + 17h + 10
```

LSTM at hidden=64, input=5: 18,826 (PRD § 5.3 v1.01).

Solving `h^2 + 17h + 10 = 18,826` ⇒ `h ≈ 128.94`. Rounding **down** to `hidden_RNN = 128`:

| hidden | RNN params | % of LSTM (18,826) |
| ---: | ---: | ---: |
|  64 |  5,194 |  27.6 % |
| **128** | **18,570** | **98.64 %** |
| 129 | 18,844 | 100.10 % |

`hidden=128` gives `18,570 params`, **98.64 % of LSTM** — a 1.36 % shortfall. The PRD guidance is "within 5 %"; we are well inside that band, and the rounded-down number is the conservative choice (any residual gap that *favours* LSTM in EXP-008 cannot be blamed on RNN-128 having extra capacity).

## Sweep

- **Model:** RNN with `hidden = 128`, all other config defaults (1 layer, tanh, default init).
- **β:** `pi/4` (matches EXP-003 — same regime where the original comparison was informative).
- **Seeds:** `1337, 1338, 1339` (matches EXP-003 / ADR-007).
- **Override surface:** `ExperimentSpec.overrides = {"signal.noise.beta": "pi/4", "model.rnn.hidden": 128}`.

**Total runs:** 3.

## Acceptance gate

The ablation is interpretable when:
- All three runs complete in under 30 minutes each (ADR-007 budget).
- Per-cell std at hidden=128 is comparable to EXP-003's RNN std at hidden=64 (i.e. capacity change does not blow up training stability).
- Mean-vs-mean comparison RNN-128 vs LSTM-64 is reported alongside the per-cell std for honest uncertainty bands.

There is **no a-priori "pass/fail"** — both H0 and H1 outcomes are acceptable; the verdict is for the collaborator session.

## What this ablation does NOT do

- It does not change the LSTM. Comparison is RNN-128 (this experiment) vs LSTM-64 (from EXP-003).
- It does not change β. β=π/4 is held fixed so EXP-008 is directly comparable to EXP-003.
- It does not tune any other hyperparameter (lr, batch, epochs, patience).
- It does not write the README "Thesis evaluation" verdict — that's the joint session that follows.

## Results

Run 2026-05-03. 3 runs in **53.6 s** wall-clock. All finite. Verified actual `n_params = 18,570` per run (matches the math; 98.64 % of LSTM-64).

### Raw runs

| seed | elapsed (s) | best_epoch / epochs_run | best_val_mse | overall_test_mse |
| --- | ---: | ---: | ---: | ---: |
| 1337 | 26.1 | 14 / 20 | 0.2667 | 0.2686 |
| 1338 | 14.1 | 13 / 19 | 0.2734 | 0.2824 |
| 1339 | 13.2 | 13 / 19 | 0.2660 | 0.2719 |

### Per-cell MSE (mean ± std, N=3 seeds)

Stacked alongside EXP-003's RNN-64 and LSTM-64 for direct comparison:

| Cell | RNN-64 (EXP-003) | **RNN-128 (EXP-008)** | LSTM-64 (EXP-003) |
| --- | --- | --- | --- |
| 2 Hz   | 0.3191 ± 0.0087 | **0.3172 ± 0.0060** | 0.3143 ± 0.0081 |
| 10 Hz  | 0.3209 ± 0.0228 | **0.3216 ± 0.0238** | 0.3183 ± 0.0240 |
| 50 Hz  | 0.2817 ± 0.0164 | **0.2798 ± 0.0154** | 0.2761 ± 0.0141 |
| 200 Hz | 0.1774 ± 0.0064 | **0.1768 ± 0.0096** | 0.1730 ± 0.0083 |

### Overall test MSE means

| Model | params | mean ± std |
| --- | ---: | --- |
| RNN-64  (EXP-003) |  5,194 | 0.2752 ± 0.0057 |
| **RNN-128 (EXP-008)** | **18,570** | **0.2743 ± 0.0072** |
| LSTM-64 (EXP-003) | 18,826 | 0.2709 ± 0.0074 |

RNN-128 is fractionally better than RNN-64 on the overall mean (Δ = 0.0009 ≈ 0.13 σ — noise) and remains worse than LSTM-64 (Δ = 0.0034 ≈ 0.45 σ).

### LSTM-64 vs RNN-128 rel(k)

`rel(k) = (MSE_RNN-128,k − MSE_LSTM-64,k) / MSE_RNN-128,k × 100` — positive ⇒ LSTM better.

| Cell | RNN-128 | LSTM-64 | **rel(k)** |  EXP-003 (RNN-64) reference |
| --- | ---: | ---: | ---: | ---: |
| 2 Hz   | 0.3172 | 0.3143 | **+0.91 %** | +1.50 % |
| 10 Hz  | 0.3216 | 0.3183 | **+1.02 %** | +0.81 % |
| 50 Hz  | 0.2798 | 0.2761 | **+1.32 %** | +2.00 % |
| 200 Hz | 0.1768 | 0.1730 | **+2.15 %** | +2.48 % |

Sign is positive at every cell — LSTM remains cell-by-cell better than the parameter-matched RNN. Magnitudes shrink at 2 Hz, 50 Hz, 200 Hz; widen at 10 Hz. Maximum rel(k) drops from +2.48 % (RNN-64) to +2.15 % (RNN-128); minimum changes very little. **The gap is preserved at this seed budget**, although every cell's |rel(k)| is well below the 10 % practical-significance threshold.

### Capacity check (sanity)

Increasing hidden 64 → 128 changes the RNN's overall mean from 0.2752 to 0.2743 — a 0.3 % improvement. Per-cell std at hidden=128 is comparable to hidden=64 (no training-stability blow-up). RNN-128 trains fine; the lack of MSE improvement is informative on its own (the bottleneck is **not** RNN capacity at this β).

## Verdict

**Deferred. Returning to collaborator** for the joint reading. Pre-meeting framing options:

- **H0 (gap was capacity)** would have predicted RNN-128 closing on LSTM. Observed: it didn't.
- **H1 (gating contributes)** is consistent with the observed numbers, but the magnitudes are small and within the seed-std band — the evidence is *suggestive, not definitive at N=3*.

## Artefacts

- `results/EXP-008-rnn-param-matched-h128/grid_log.json` — 3-run record.
- `results/EXP-008-rnn-param-matched-h128/summary.json` — derived per-cell mean/std + LSTM-vs-RNN-128 rel(k) and EXP-003 cross-comparison.
- `results/EXP-008-rnn-param-matched-h128/<run_dir>/` per seed.

## Artefacts

- `results/EXP-008-rnn-param-matched-h128/<run_dir>/` per seed.
- `results/EXP-008-rnn-param-matched-h128/grid_log.json` — machine-readable record.
- `results/EXP-008-rnn-param-matched-h128/summary.json` — derived per-cell mean/std + LSTM-vs-RNN-128 rel(k).

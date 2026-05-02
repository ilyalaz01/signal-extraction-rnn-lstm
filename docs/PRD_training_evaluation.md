# PRD — Training & Evaluation

> **Document version:** 1.01
> **Status:** Approved.
> **Owns:** the training loop (`services/training.py`), the evaluation metrics (`services/evaluation.py`), the experiment-spec / result types (`sdk/sdk.py`), the `results.json` schema, the thesis evaluation protocol, and the test plan for these components.
> **Companion:** `HOMEWORK_BRIEF.md` § 6 / § 9 / § 10, `docs/PRD.md` v1.01, `docs/PLAN.md` v1.02 § 6.1 / § 7 / § 9, `docs/PRD_dataset_construction.md` v1.01 (SplitDatasets / DataLoader contract), `docs/PRD_models.md` v1.01 (SignalExtractor / ModelKind), `docs/adr/ADR-013-config-override-policy.md` (deferred), `docs/adr/ADR-014-results-layout.md` (M0, written), `docs/adr/ADR-007-seeds-per-cell.md` (deferred).
>
> **Changelog from v1.00:** § 8.4 Outcome A monotonicity criterion replaced with Spearman-ρ-based test from § 8.6 (was a strict inequality chain — too brittle for a single-seed run). Outcome B reworded as a catch-all to close the coverage gap where `rel(s_2) ≥ 10%` but `rel(s_1) < 10%` fit no outcome. § 8.3 threshold rationale strengthened with seed-std re-validation procedure. § 7.3 cross-frequency energy disclaimer added. § 4.3 pseudocode corrected — final-epoch weights now captured before restoring best (prior version would have written best weights into `checkpoint_final.pt`). T-TR-02 assertion strengthened (50 epochs / 50% reduction, matching PRD_models trainability smokes). T-TR-11 added to verify the § 4.3 fix.

This is the dedicated PRD for the **training and evaluation** subsystem. It defines the loss function, optimizer, training protocol, checkpointing, the per-frequency MSE evaluation, and — uniquely among the four dedicated PRDs — the **thesis evaluation protocol**: the bridge from a numerical per-frequency MSE table to the README's "Thesis evaluation" section.

Out of scope: signal generation (`PRD_signal_generation.md`), window sampling (`PRD_dataset_construction.md`), model architectures (`PRD_models.md`).

---

## 1. Purpose

Given a `SignalExtractor` model and a `SplitDatasets`, the training subsystem:

- Runs the training loop (Adam, MSE loss, early stopping on val MSE).
- Saves checkpoints and a per-epoch log to the run directory (ADR-014 layout).
- Returns a `TrainingResult` carrying the best-checkpoint weights and the epoch history.

The evaluation subsystem then:

- Computes overall test MSE and a per-frequency MSE breakdown on the test split.
- Writes `results.json` (schema: § 10.4).
- Returns an `EvalResult` that the analysis notebook and README "Thesis evaluation" section consume.

The sensitivity runner (owned by the SDK layer, typed in this PRD) wraps train + evaluate over a grid of `ExperimentSpec`s, parametrizing over model kind, noise parameters, and seed.

---

## 2. Loss Function

### 2.1 Formula

$$\mathcal{L} = \frac{1}{B \cdot W} \sum_{i=1}^{B} \sum_{j=1}^{W} \left(\hat{y}_{ij} - y_{ij}\right)^2$$

where $B$ is batch size and $W = 10$ is the output window length. This matches `HOMEWORK_BRIEF.md` § 6 (mean over examples, squared L2 norm per example) and is identical to `torch.nn.MSELoss(reduction='mean')`, which averages over all elements jointly.

### 2.2 Implementation

```python
criterion = torch.nn.MSELoss(reduction='mean')
loss = criterion(y_pred, y_target)   # y_pred, y_target: (B, 10), float32
```

`reduction='mean'` is pinned explicitly in code and tested (T-TR-01). The alternative `reduction='sum'` would make the loss scale with batch size and break the learning rate's meaning across batch-size ablations. `reduction='none'` is used only in evaluation (§ 7.2) to compute per-example losses before grouping by frequency.

---

## 3. Optimizer & Learning Rate

### 3.1 Default (v1.00)

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
)
```

All parameters are read from `config/setup.json § training` (no hard-coded values); the table below records defaults and constraints.

| Parameter | Default | Source / Rationale |
| --- | --- | --- |
| `lr` | `1e-3` | Standard Adam default; safe for small MSE regression tasks on normalized-ish signals (O(1) amplitude). |
| `betas[0]` | `0.90` | PyTorch / paper default for first-moment decay. |
| `betas[1]` | `0.999` | PyTorch / paper default for second-moment decay. |
| `eps` | `1e-8` | PyTorch default; no denominator-zero risk at float32 with typical gradients. |
| `weight_decay` | `0.0` | No L2 regularization in v1.00. FC:param ratio ≈ 5.3×; RNN ≈ 5.8×; LSTM ≈ 1.6×. LSTM is closest to the overfitting boundary but the task is a simple sinusoidal regression — overfitting risk is low and regularization would cloud the thesis comparison. Reopen if `EXP-001` shows a persistent train/val gap for LSTM. |

### 3.2 Learning rate schedule

**None in v1.00.** `config.training.scheduler = null`. Fixed lr for the full run.

**Alternatives considered:**
- *`ReduceLROnPlateau(factor=0.5, patience=3, mode='min')`*. Natural fit; reduces lr on val-MSE plateau. Rejected for v1.00 — an additional hyperparameter would interact with `early_stop_patience` in non-obvious ways and complicate reproducing the baseline. Reopen in a new ADR if `EXP-001` shows slow late-epoch convergence.
- *Cosine annealing*. Requires knowing `T_max`; overkill for a 30-epoch run.

---

## 4. Training Protocol

### 4.1 Configuration (from `config/setup.json § training`)

| Key | Type | Default | Constraint |
| --- | --- | --- | --- |
| `batch_size` | int | `256` | > 0; ≤ `n_train`. |
| `epochs` | int | `30` | ≥ 1. |
| `early_stop_patience` | int | `5` | ≥ 1. 0 disables early stopping. |
| `optimizer` | str | `"adam"` | `"adam"` only in v1.00; `ValueError` otherwise. |
| `lr` | float | `1e-3` | > 0. |
| `scheduler` | str\|null | `null` | `null` only in v1.00; `ValueError` otherwise. |

### 4.2 Early stopping

- **Mode:** min on validation MSE.
- **Patience:** stop training if val MSE does not strictly improve (decrease) for `early_stop_patience` consecutive epochs.
- **Best tracking:** the best-val-MSE epoch's `model.state_dict()` is written to `checkpoint_best.pt` at the moment of improvement. If training ends by patience, the returned `TrainingResult.model` has the best-checkpoint weights loaded back — **not** the final-epoch weights.
- **Delta:** any strict improvement (no `min_delta` in v1.00). See § 15 for the alternative with `min_delta > 0`.

### 4.3 Training step

```text
for each epoch:
    model.train()
    for each batch (selector, w_noisy, w_clean) from train DataLoader:
        optimizer.zero_grad()
        y_pred = model(selector, w_noisy)      # (B, 10)
        loss   = criterion(y_pred, w_clean)    # scalar
        loss.backward()
        optimizer.step()
    train_mse = mean loss over all batches (accumulated, not re-forward-passed)

    model.eval()
    with torch.no_grad():
        val_mse = evaluate on val DataLoader (same criterion)

    log EpochResult(epoch, train_mse, val_mse, elapsed_s)
    if val_mse improves: save checkpoint_best.pt
    if patience exhausted: break

# capture final-epoch weights BEFORE restoring best
final_state = {k: v.clone() for k, v in model.state_dict().items()}
save checkpoint_final.pt with final_state, current epoch, current val_mse
restore best weights: model.load_state_dict(torch.load(checkpoint_best.pt)["model_state_dict"])
```

`checkpoint_final.pt` must be written from `final_state` (the last-epoch weights), not from the model after `load_state_dict` — that order would silently give it best-epoch weights. This is verified by T-TR-11.

---

## 5. DataLoader & Reproducibility

### 5.1 DataLoader construction

```python
train_gen = torch.Generator().manual_seed(dataloader_seed)
train_dl  = DataLoader(datasets.train, batch_size=B, shuffle=True,
                       generator=train_gen, num_workers=0)
val_dl    = DataLoader(datasets.val,   batch_size=B, shuffle=False, num_workers=0)
test_dl   = DataLoader(datasets.test,  batch_size=B, shuffle=False, num_workers=0)
```

`num_workers=0` is load-bearing for determinism (per `PLAN.md` § 11.4 and `config.runtime.dataloader.num_workers`). Val and test use no generator because they are not shuffled.

### 5.2 Seed derivation

The training loop receives `dataloader_seed` as a parameter; the SDK derives it before calling `services.training.train()`:

```python
dataloader_seed = seeding.derive(runtime_seed, "train_dataloader")
```

`seeding.derive` uses `np.random.SeedSequence(runtime_seed).spawn(1)` with a string key hashed to an integer — the same mechanism as `PRD_dataset_construction.md` § 6. This keeps the DataLoader shuffle reproducible and independent of other RNG consumers (the model init seed, the corpus sampling seed).

### 5.3 What is deterministic

Given the same `(model kind, runtime_seed, config)`:
- Model weight initialization (`torch.manual_seed` set by `shared.seeding` before `models.build()`).
- Corpus generation (corpus seed derived from runtime seed).
- Window sampling and split assignment (sampling seed derived from runtime seed).
- DataLoader batch order (dataloader seed derived from runtime seed).
- Adam optimizer state (deterministic given same weights and same batches, no internal randomness).

The only non-determinism risk is `torch.use_deterministic_algorithms(True)` enforcement — already required by `shared.seeding` (`PLAN.md` § 11.1). If a CUDA op lacks a deterministic implementation, `seeding.py` falls back to disabling CUDA and logging a warning; verified by the reproducibility integration test `T-IT-02`.

---

## 6. Checkpointing & Logging

### 6.1 Run directory (ADR-014)

```text
results/<utc_timestamp>__<model_kind>__<seed>/
    checkpoint_best.pt        # best val MSE state dict + metadata
    checkpoint_final.pt       # final epoch state dict + metadata
    train.log                 # epoch table (tab-separated)
    results.json              # full result record (schema: § 10.4)
```

`<utc_timestamp>` is `datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")`. This format sorts lexicographically by time, so `ls results/` gives chronological order. `<seed>` is the integer runtime seed.

### 6.2 Checkpoint format

```python
torch.save({
    "epoch":    best_epoch,
    "val_mse":  best_val_mse,
    "model_state_dict": model.state_dict(),
}, run_dir / "checkpoint_best.pt")
```

`checkpoint_final.pt` has the same schema with the final epoch's values. The `model_state_dict` key is intentionally named verbatim so that `torch.load(...)["model_state_dict"]` is self-documenting at load time.

### 6.3 Epoch log

`train.log` is a tab-separated table, one header row then one row per epoch:

```
epoch	train_mse	val_mse	elapsed_s
1	0.2341	0.2289	1.23
2	0.1874	0.1801	1.21
...
```

Written incrementally (one row per epoch, flushed), not buffered, so a crashed run leaves a partial but readable log.

---

## 7. Evaluation Metrics

### 7.1 Overall test MSE

Computed on the test split using the best-checkpoint model (weights already restored in `TrainingResult.model`):

```python
overall_test_mse = mean MSE over all test examples
```

Single scalar. Useful for gross model ranking but not the thesis metric.

### 7.2 Per-frequency MSE (headline metric)

For each sinusoid index $k \in \{0, 1, 2, 3\}$, restrict the test set to examples where `selector.argmax() == k` and compute MSE:

```python
per_freq_mse: dict[int, float]   # k → MSE on test examples for sinusoid k
```

```text
mask_k    = (selector.argmax(dim=-1) == k)          # (B,) bool
mse_k     = F.mse_loss(y_pred[mask_k], y_target[mask_k], reduction='mean')
```

Because sampling is iid and balanced (25% per channel by construction — `PRD_dataset_construction.md` § 4), each group has ≈ 937 test examples (3 750 / 4). This is enough for a stable MSE estimate.

The four frequency labels are `constants.FREQUENCIES_HZ = [2, 10, 50, 200]` (Hz). `EvalResult` exposes a `per_freq_hz` property mapping Hz → MSE for notebook convenience.

### 7.3 No normalization of metrics

MSE is reported in **raw signal units squared** (amplitude² ≈ V² for unit-amplitude sinusoids). No normalization by signal power or baseline MSE. Reasons:
- The signals have equal amplitude (A_k = 1.0 for all k in v1.00 default), so cross-frequency MSE comparisons are on the same scale.
- Normalized metrics (e.g., "% of naive predictor MSE") add a computation that requires defining the naive predictor — not specified by the lecturer. Raw MSE is interpretable and grader-safe.
- If amplitudes are changed in an ablation, raw MSE becomes cross-frequency incomparable; this is a known limitation, documented in the analysis notebook.

**Caveat — cross-frequency signal energy.** Although all four sinusoids have equal amplitude (A_k = 1.0), they carry different signal energy within a 10-sample window: a 200 Hz sinusoid completes 2 full cycles, so the RMS energy over 10 ms is similar to the continuous RMS; a 2 Hz sinusoid covers ~2% of a cycle, so the window looks nearly linear with very low local variance. MSE therefore probes structurally different regression problems per frequency — low-frequency windows are easier (nearly constant slope) and high-frequency windows are harder (full oscillation to track). The 10% relative threshold (§ 8.3) is robust to this asymmetry because it is a **within-frequency comparison** (LSTM vs RNN at the same k); the cross-frequency monotonicity test in § 8.4 / § 8.6 is also robust for the same reason. Raw MSE numbers **must not be compared across frequencies without caveat** — the analysis notebook must include this note in the figure caption for the per-frequency MSE bar chart.

---

## 8. Thesis Evaluation Protocol

This section is the bridge from per-frequency MSE numbers to the README's "Thesis evaluation" section. It is specification, not commentary — the analysis notebook and README must follow this protocol exactly.

### 8.1 The thesis (from HOMEWORK_BRIEF.md § 2)

> RNN is good for short-term memory → high-frequency components. LSTM maintains long-term dependencies → low-frequency components.

Operationally: LSTM achieves lower test MSE than RNN on low-frequency sinusoids; RNN approaches FC on high-frequency sinusoids (where short-term structure is sufficient and recurrence adds little).

### 8.2 Comparison table (produced by analysis notebook, EXP-001)

The notebook must produce the following table for each pair `(model_A, model_B)` from `{FC, RNN, LSTM}`:

| Sinusoid | Freq (Hz) | FC MSE | RNN MSE | LSTM MSE | LSTM vs RNN (%) | Prediction | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| s_1 | 2 | — | — | — | — | LSTM best | — |
| s_2 | 10 | — | — | — | — | LSTM ≥ RNN | — |
| s_3 | 50 | — | — | — | — | mixed | — |
| s_4 | 200 | — | — | — | — | RNN ≈ FC | — |

**"LSTM vs RNN (%)"** column is the relative improvement:

$$\text{rel}(k) = \frac{\text{MSE}_{\text{RNN},k} - \text{MSE}_{\text{LSTM},k}}{\text{MSE}_{\text{RNN},k}} \times 100$$

Positive = LSTM better. Negative = RNN better.

### 8.3 Practical significance threshold

A difference at sinusoid $k$ is **practically significant** if $|\text{rel}(k)| \geq 10\%$.

10% is a provisional default, chosen to be (a) substantially larger than typical seed-to-seed standard deviation in MSE on this scale of regression task and (b) low enough not to dismiss moderate architectural effects. After the multi-seed grid (ADR-007), the threshold should be re-validated as ≥ 2× the observed seed-std for the most-variable model. If the observed seed-std exceeds 5%, this PRD is reopened and the threshold raised. The threshold must be fixed before EXP-001 runs — the protocol must not be retroactively adjusted to fit results.

### 8.4 Outcome classification

Evaluate the following decision tree on the EXP-001 results table:

**Outcome A — Full confirmation:**
- `rel(s_1) ≥ 10%` AND `rel(s_2) ≥ 10%` (LSTM significantly better on both low-freq sinusoids), AND
- `|rel_FC_RNN(s_4)| < 10%` where `rel_FC_RNN(k) = (MSE_RNN,k - MSE_FC,k) / MSE_FC,k` (RNN not significantly better than FC at 200 Hz), AND
- Spearman ρ between `FREQUENCIES_HZ` and `{rel(k)}` is **negative** AND p-value < 0.10 (one-sided H₁: ρ < 0 — the monotonicity test from § 8.6, now load-bearing for Outcome A). This replaces the strict inequality chain `rel(s_1) ≥ rel(s_2) ≥ ...` from v1.00, which would falsely demote a clear trend on a single-seed run where one pair flips by chance.

**Outcome B — Partial confirmation:**
- Not Outcome A (full confirmation), AND
- Not Outcome C (refutation), AND
- Not Outcome D (capacity confound).
- i.e., some evidence for the thesis is present but not strong enough for full confirmation. Examples: `rel(s_1) ≥ 10%` but `rel(s_2) < 10%`; or both low-freq thresholds met but Spearman ρ is not significant; or the low-freq LSTM advantage is present but RNN is not competitive with FC at high frequency.

**Outcome C — Refutation:**
- `rel(s_1) < 10%` AND `rel(s_2) < 10%` — LSTM does not outperform RNN at low frequencies by a practically significant margin.

**Outcome D — Capacity confound:**
- `rel(k) ≥ 10%` for ALL k ∈ {0,1,2,3}. LSTM wins uniformly across all frequencies.
- **Trigger:** run `EXP-008` (parameter-matched RNN at hidden ≈ 128). If the confound disappears in EXP-008, attribute the uniform LSTM win to parameter count, not gating. If it persists, the thesis is strengthened (gating helps even when capacity is matched).

### 8.5 README template sentence

The README "Thesis evaluation" section must contain one sentence per sinusoid in this form, filled from the EXP-001 table:

> At [freq] Hz (sinusoid s_{k+1}), LSTM achieves [|rel(k)|]% [lower / higher] test MSE than RNN ([lstm_mse:.4f] vs [rnn_mse:.4f]), [within / above] the 10% practical significance threshold. FC achieves [fc_mse:.4f]. **[Outcome label: "Consistent with full confirmation" / "Partially consistent" / "Inconsistent with" / "Confounded by parameter asymmetry — see EXP-008"] of the lecturer's thesis.**

And an overall verdict line:

> **Overall thesis verdict (EXP-001, seed [seed], single run):** [Full confirmation / Partial confirmation / Refutation / Capacity confound — see EXP-008]. Multi-seed statistical tests deferred to ADR-007 (post-M5).

The "single run" qualifier is mandatory in v1.00: one seed is insufficient for a confidence interval. ADR-007 defines the multi-seed procedure.

### 8.6 Monotonicity test

The analysis notebook must also report the Spearman rank correlation between `FREQUENCIES_HZ` and `{rel(k)}`. A negative ρ (LSTM advantage decreases as frequency increases) is quantitative support for the thesis trend even if the 10% threshold is not crossed at every point. Report ρ and the one-sided p-value (H₁: ρ < 0). This test is load-bearing for Outcome A (§ 8.4) — it is not a side analysis.

With four data points (k = 0,1,2,3), the test has low power; a significant result at p < 0.10 requires a near-perfect monotone ordering. This is intentionally strict for Outcome A. An insignificant ρ with `rel(s_1) ≥ 10%` and `rel(s_2) ≥ 10%` still qualifies as Outcome B (partial confirmation).

---

## 9. Sensitivity & Ablation Runner

### 9.1 ExperimentSpec

```python
@dataclass(frozen=True)
class ExperimentSpec:
    """
    Input:  (model_kind, optional seed override, optional config overrides).
    Output: identifies one training run for SDK.run_experiment().
    Setup:  overrides is a flat dict of dotted-path → value pairs applied
            to setup.json before building the corpus.
            Examples: {"signal.noise.alpha": 0.10, "signal.noise.beta": "pi"}.
    """
    model_kind: ModelKind
    seed:       int | None = None          # None → use config.runtime.seed
    overrides:  dict[str, Any] = field(default_factory=dict)
```

`ExperimentSpec` lives in `sdk/sdk.py` alongside `ExperimentResult`. It is not a service-layer type.

### 9.2 ExperimentResult

```python
@dataclass(frozen=True)
class ExperimentResult:
    """
    Input:  produced by SDK.run_experiment(spec).
    Output: this record.
    Setup:  train_result.model carries best-checkpoint weights.
    """
    spec:         ExperimentSpec
    train_result: TrainingResult
    eval_result:  EvalResult
```

### 9.3 Standard sensitivity grid (EXP-002 through EXP-004 — M5)

The baseline single experiment (EXP-001) runs all three models at default config. The sensitivity grid parametrizes one dimension at a time (OAT approach per `SOFTWARE_PROJECT_GUIDELINES.md` § 8.1):

| Dimension | Values to sweep | Fixed |
| --- | --- | --- |
| `signal.noise.alpha` | 0.01, 0.05 (default), 0.10, 0.20 | all others default |
| `signal.noise.beta` | `"0"`, `"pi/4"`, `"pi"`, `"2*pi"` (default) | all others default |
| `runtime.seed` | 3–5 seeds (exact count: ADR-007) | model=all three, others default |

The SDK method `SDK.run_grid(specs: list[ExperimentSpec]) -> list[ExperimentResult]` executes the list sequentially (no parallelism in v1.00 — guidelines § 14 is not engaged for a single-process local run).

### 9.4 Config override mechanics

`SDK.run_experiment(spec)` applies `spec.overrides` to the loaded config using `shared.config.apply_overrides(config_dict, overrides)`:

```python
def apply_overrides(cfg: dict, overrides: dict[str, Any]) -> dict:
    """Applies dotted-path overrides; raises KeyError on unknown paths."""
```

Deep-merge, not shallow. Unknown paths raise `KeyError` immediately (fail fast). The modified config is not written to disk — it is only used for the duration of the experiment. This behaviour is tested by T-TR-09 and will be formally locked in ADR-013 when the override policy is finalized.

---

## 10. Public API & Type Definitions

### 10.1 `services/training.py` public surface

```python
@dataclass(frozen=True)
class TrainingConfig:
    """
    Input:  parsed from config["training"].
    Output: controls the training loop.
    Setup:  all fields validated at construction time.
    """
    batch_size:           int
    epochs:               int
    early_stop_patience:  int
    optimizer:            str    # "adam" only in v1.00
    lr:                   float
    scheduler:            str | None  # None only in v1.00

@dataclass
class EpochResult:
    """
    Input:  produced once per epoch by _run_epoch().
    Output: row in train.log and train_history list.
    Setup:  mutable because it is accumulated into a list.
    """
    epoch:      int
    train_mse:  float
    val_mse:    float
    elapsed_s:  float

@dataclass
class TrainingResult:
    """
    Input:  produced by train().
    Output: carries the best-checkpoint model + epoch history.
    Setup:  model has best-checkpoint weights loaded (not final-epoch weights).
            Not frozen because nn.Module is mutable.
    """
    model:         SignalExtractor   # best-checkpoint weights
    model_kind:    ModelKind
    train_history: list[EpochResult]
    best_epoch:    int
    best_val_mse:  float
    run_dir:       Path

def train(
    model:             SignalExtractor,
    datasets:          SplitDatasets,
    config:            TrainingConfig,
    run_dir:           Path,
    dataloader_seed:   int,
) -> TrainingResult:
    """
    Input:  model (weights at PyTorch defaults), datasets, config, run_dir, seed.
    Output: TrainingResult with best-checkpoint weights.
    Setup:  writes checkpoint_best.pt, checkpoint_final.pt, train.log to run_dir.
            run_dir must exist before call (created by SDK).
    """
```

### 10.2 `services/evaluation.py` public surface

```python
@dataclass(frozen=True)
class EvalResult:
    """
    Input:  produced by evaluate().
    Output: carries overall and per-frequency test MSE.
    Setup:  per_freq_mse keys are sinusoid indices 0..3.
            per_freq_hz property maps to Hz labels from constants.FREQUENCIES_HZ.
    """
    overall_test_mse: float
    per_freq_mse:     dict[int, float]   # sinusoid index k → MSE
    run_dir:          Path

    @property
    def per_freq_hz(self) -> dict[float, float]:
        """Maps FREQUENCIES_HZ[k] → per_freq_mse[k] for notebook convenience."""
        ...

def evaluate(
    model:    SignalExtractor,
    datasets: SplitDatasets,
    run_dir:  Path,
) -> EvalResult:
    """
    Input:  model with best-checkpoint weights, datasets (test split used),
            run_dir for results.json write.
    Output: EvalResult with overall and per-frequency test MSE.
    Setup:  model set to eval() internally; torch.no_grad() context used.
            Writes results.json to run_dir.
    """
```

### 10.3 SDK surface (additions to `PLAN.md` § 6.1)

The SDK's existing `train()` and `evaluate()` methods are wired to the service functions above. Two additions:

```python
def run_experiment(self, spec: ExperimentSpec) -> ExperimentResult:
    """
    Input:  ExperimentSpec (model_kind, optional seed override, config overrides).
    Output: ExperimentResult (spec + TrainingResult + EvalResult).
    Setup:  applies overrides, creates run_dir, calls train() then evaluate().
    """

def run_grid(self, specs: list[ExperimentSpec]) -> list[ExperimentResult]:
    """
    Input:  list of ExperimentSpecs.
    Output: list of ExperimentResults, one per spec, in input order.
    Setup:  sequential execution; no parallelism in v1.00.
    """
```

### 10.4 `results.json` schema

```jsonc
{
  "version": "1.00",
  "spec": {
    "model_kind": "fc",
    "seed": 1337,
    "overrides": {}
  },
  "training": {
    "best_epoch":    12,
    "best_val_mse":  0.004312,
    "epochs_run":    17,
    "stopped_early": true
  },
  "evaluation": {
    "overall_test_mse": 0.005104,
    "per_freq_mse": {
      "0": 0.008021,
      "1": 0.006013,
      "2": 0.004187,
      "3": 0.002195
    }
  }
}
```

Keys in `per_freq_mse` are string representations of sinusoid indices (JSON requires string keys). The analysis notebook maps `"0"` → `constants.FREQUENCIES_HZ[0]` = 2 Hz.

---

## 11. Module Layout & LOC Budget

| File | PLAN.md ceiling | Internal target |
| --- | --- | --- |
| `services/training.py` | ~140 | ~115 |
| `services/evaluation.py` | ~130 | ~90 |
| Additions to `sdk/sdk.py` | (within 120 total) | ~30 additional lines |

Both ceilings will be confirmed in the PLAN.md v1.02 sweep after this PRD is approved. If either file grows past 130 LOC during implementation, the split strategy is: extract `_dataloader_factory.py` from `training.py` (DataLoader construction + seed wiring) and `_results_writer.py` from `evaluation.py` (results.json write + figure hooks).

---

## 12. Test Specification

All training tests live in `tests/unit/test_training.py`; all evaluation tests in `tests/unit/test_evaluation.py`. The end-to-end smoke lives in `tests/integration/test_sdk_smoke.py` (extends the existing SDK smoke from `PRD_dataset_construction.md` AC-DS-7).

### 12.1 Unit tests — training

| ID | Behavior | Assertion |
| --- | --- | --- |
| T-TR-01 | Loss uses `reduction='mean'` | manually compute `((y_hat - y)**2).mean()` on a batch; compare to `criterion(y_hat, y)` — equal to 6 decimal places. |
| T-TR-02 | Loss decreases substantially | after 50 epochs of full-batch SGD on a fixed 4-example mini-set, `train_mse` is at least 50% lower than `initial_mse` (measured before any gradient step). Aligns with the trainability smokes in `PRD_models.md` T-MD-10..12. |
| T-TR-03 | Early stopping fires | construct a mock val-MSE sequence that does not improve for `patience + 1` epochs; assert training stops at `patience` epochs after last improvement. |
| T-TR-04 | Best checkpoint saved on improvement | val MSE improves at epoch 3, does not at epochs 4–7; `checkpoint_best.pt` contains `{"epoch": 3, ...}`. |
| T-TR-05 | Best weights restored | `TrainingResult.model` state dict == weights from `checkpoint_best.pt`, not `checkpoint_final.pt`, when they differ. |
| T-TR-06 | `TrainingResult` invariants | `best_epoch ∈ [0, len(train_history)-1]`; `best_val_mse == train_history[best_epoch].val_mse`; `run_dir` exists and contains `train.log`. |
| T-TR-07 | `train.log` format | file has header row + one row per epoch run; epoch column is monotone; MSE columns are positive floats. |
| T-TR-08 | Determinism | two `train()` calls with the same `(model, datasets, config, dataloader_seed)` and same global seed produce identical `train_history` lists. |
| T-TR-09 | Config override applied | `apply_overrides({"signal.noise.alpha": 0.20}, ...)` changes alpha and propagates to corpus; unknown path raises `KeyError`. |
| T-TR-10 | `run_dir` follows ADR-014 pattern | name matches `r"\d{8}T\d{6}Z__[a-z]+__\d+"`. |
| T-TR-11 | `checkpoint_final.pt` holds actual final-epoch weights | training fixture: best at epoch 3, final at epoch 8 (patience not reached); assert `checkpoint_final.pt` state dict differs from `checkpoint_best.pt` state dict (at least one parameter tensor unequal). Pins the § 4.3 save-order fix. |

### 12.2 Unit tests — evaluation

| ID | Behavior | Assertion |
| --- | --- | --- |
| T-EV-01 | Output types | `overall_test_mse` is `float`; `per_freq_mse` is `dict[int, float]` with keys `{0,1,2,3}`. |
| T-EV-02 | Untrained model gives non-trivial MSE | `overall_test_mse > 0` and finite. |
| T-EV-03 | Per-frequency correctness | manually filter test batch by `selector.argmax() == k`; compute MSE with `F.mse_loss(..., reduction='mean')`; matches `per_freq_mse[k]` to 6 decimal places. |
| T-EV-04 | `per_freq_hz` property | `eval_result.per_freq_hz[2.0]` == `per_freq_mse[0]`; `per_freq_hz[200.0]` == `per_freq_mse[3]`. |
| T-EV-05 | `results.json` written | file exists at `run_dir / "results.json"`; loads as valid JSON; top-level keys are `{"version","spec","training","evaluation"}`; `per_freq_mse` has 4 string-keyed entries. |
| T-EV-06 | Perfect model → zero MSE | a model that returns `w_clean` directly (oracle); `per_freq_mse[k] == 0.0` for all k. |

### 12.3 Integration test — end-to-end smoke

| ID | Behavior | Assertion |
| --- | --- | --- |
| T-IT-01 | Signal → dataset → model → 1 epoch → metric | for each of `["fc", "rnn", "lstm"]`: `SDK.generate_corpus()` → `SDK.build_dataset()` → `SDK.train(kind, datasets)` (1 epoch, tiny corpus: n_train=200, n_val=50, n_test=50) → `SDK.evaluate(result, datasets)`. Asserts `overall_test_mse > 0`, `isfinite`, `per_freq_mse` has 4 keys, `run_dir` contains all four files. Runs in < 60 s on CPU. |
| T-IT-02 | Reproducibility | same spec, same seed, two `SDK.run_experiment()` calls → `train_history` lists are element-wise identical; `overall_test_mse` equal to 8 decimal places. |
| T-IT-03 | Grid runner | `SDK.run_grid([spec_fc, spec_rnn, spec_lstm])` returns 3 `ExperimentResult`s; each `run_dir` is distinct and exists. |

### 12.4 Coverage target

`services/training.py` and `services/evaluation.py` together are targeted at ≥ 85% line coverage (global floor per `SOFTWARE_PROJECT_GUIDELINES.md` § 5.2). Both modules are short and pure; ≥ 90% is achievable without heroics.

---

## 13. Edge Cases

| Case | Behavior |
| --- | --- |
| `early_stop_patience = 0` | Disabled: training runs for exactly `epochs` epochs. `best_epoch` = epoch with minimum val MSE over the full run. |
| `epochs = 1` | One epoch, no early stopping possible. `best_epoch = 0`. Both checkpoints written (identical content because final and best are the same epoch). |
| `n_val` examples all map to one frequency | `per_freq_mse` at that frequency computed; other keys have MSE from 0 test examples → `nan`. The 3750-example balanced test set makes this impossible in v1.00, but a future config with `n_test = 4` could hit it. Guard: if a group is empty, `per_freq_mse[k] = float("nan")` with a `logging.warning`. |
| NaN loss | `ValueError("NaN loss at epoch {e}, step {s}; check initialization or learning rate")` — training aborted, no checkpoint written. No silent continuation. |
| `run_dir` already exists | `FileExistsError`. The SDK creates `run_dir` with `mkdir(exist_ok=False)`. Prevents silent overwrite of a prior run. |
| Model not in `.eval()` mode during evaluation | `evaluate()` calls `model.eval()` and restores training mode (`model.train()`) on exit via context manager — not left to the caller. |
| `spec.overrides` contains a path that changes `window` or `n_channels` | Currently raises `KeyError` only if the path does not exist. A future guard should detect shape-breaking overrides (window ≠ 10, n_channels ≠ 4) and raise before corpus generation. Listed as a deferred hardening item, not a v1.00 requirement. |

---

## 14. Acceptance Criteria

The training and evaluation subsystem ships when:

- AC-TE-1: All tests T-TR-01 through T-TR-11, T-EV-01 through T-EV-06 pass.
- AC-TE-2: Integration tests T-IT-01, T-IT-02, T-IT-03 pass and T-IT-01 completes in < 60 s on CPU.
- AC-TE-3: `services/training.py` ≤ 140 LOC; `services/evaluation.py` ≤ 130 LOC.
- AC-TE-4: `ruff check` clean.
- AC-TE-5: Coverage ≥ 85% on both modules.
- AC-TE-6: Building Block docstrings on `TrainingConfig`, `EpochResult`, `TrainingResult`, `EvalResult`, `ExperimentSpec`, `ExperimentResult`, `train()`, `evaluate()`, `run_experiment()`, `run_grid()`.
- AC-TE-7: `results.json` written by each `SDK.run_experiment()` call conforms to the § 10.4 schema (validated by T-EV-05 and the integration smoke T-IT-01).
- AC-TE-8: The analysis notebook cell for EXP-001 produces the § 8.2 comparison table and the § 8.5 README template sentences (delivered in M5, not M4).

---

## 15. Alternatives Considered

- **`reduction='sum'` for MSE loss.** Rejected; loss scales with batch size, breaking lr transferability across batch-size ablations.
- **`ReduceLROnPlateau` as v1.00 schedule.** Rejected; adds a second hyperparameter that interacts with `early_stop_patience` in non-obvious ways. Reopen in a new ADR if EXP-001 shows late-epoch stagnation.
- **`min_delta > 0` for early stopping.** Rejected in v1.00; any strict improvement is enough to reset the patience counter. A non-zero `min_delta` (e.g. 1e-5) would prevent counting tiny improvements as progress, which could extend training artificially on a plateaued val loss. Reopen if the early-stopping behaviour in EXP-001 is clearly too trigger-happy.
- **`AdamW` instead of `Adam`.** Would add `weight_decay` as an implicit per-parameter decoupled L2 penalty. Rejected for the same reason as `weight_decay=0.0` in § 3.1: LSTM is the only model where overfitting is plausible, and we don't want regularization differences to pollute the comparison.
- **Gradient clipping.** Rejected. The RNN training window is only 10 steps — gradient explosions are unlikely at `tanh` + default init on this task. No clipping in v1.00; add `torch.nn.utils.clip_grad_norm_` under a config key if EXP-001 shows NaN losses.
- **Saving every-N-epoch checkpoints.** Rejected; storage overhead for only 30 epochs is not worth the complexity. Best and final are sufficient for the thesis comparison.
- **Storing `per_freq_mse` keys as Hz floats.** Rejected as primary key; float keys are fragile in JSON (JSON has no float type — all numbers deserialise as IEEE 754, and `200.0` vs `200` vs `2e2` are the same value in JSON but handled differently by different parsers). Integer indices are stable. The Hz mapping is deferred to the notebook via `constants.FREQUENCIES_HZ`.
- **Parallel grid execution.** Rejected for v1.00; three sequential runs (one per model kind) complete in minutes on CPU. Parallelism would require thread-safe logging and filesystem writes. Revisit for the multi-seed grid in M5 if total runtime exceeds 30 minutes.
- **Per-sinusoid signal-power normalization of MSE.** Rejected in § 7.3; all amplitudes are equal at default config so raw MSE is on the same scale per frequency; cross-frequency comparisons carry a caveat regardless (§ 7.3).
- **Strict inequality chain for Outcome A monotonicity** (`rel(s_1) ≥ rel(s_2) ≥ ...`). Rejected in v1.01 — see § 8.4 and changelog. Replaced with Spearman ρ test.

---

## 16. Decisions Locked Without Explicit User Input

These are calls I made during drafting. Surfaced for review; all confirmed at v1.01.

1. **`TrainingResult.model` carries in-memory best-checkpoint weights**, not a path. This means `SDK.evaluate(result, datasets)` does not need to reload from disk when called immediately after `train()`. Tradeoff: `TrainingResult` is not serializable without pickling the model. If you later prefer a path-only contract (lighter, always reloads), change `TrainingResult.model` to `best_checkpoint_path: Path` and update `SDK.evaluate()` to load from it. Confirmed: keep.

2. **Practical significance threshold = 10%** in § 8.3, with the re-validation procedure tied to ADR-007 multi-seed results. The threshold is provisional but must be fixed before EXP-001 runs. Confirmed: keep with strengthened rationale.

3. **`EvalResult` is frozen** despite `per_freq_mse` being a dict. Python `dataclass(frozen=True)` does not deep-freeze dict contents, but dict identity is frozen (you cannot reassign `eval_result.per_freq_mse = ...`). Confirmed: keep as-is, no `MappingProxyType` wrapping.

4. **Spearman ρ test** as the monotonicity check (§ 8.6). Non-parametric rank correlation — appropriate when we have four frequency points and no assumption about functional form. The one-sided H₁ (ρ < 0) is the thesis direction. Now also load-bearing for Outcome A (§ 8.4). Confirmed.

5. **`apply_overrides` raises `KeyError` on unknown paths** (§ 9.4). Safe default; unknown paths almost always indicate a typo. Confirmed.

6. **`ExperimentSpec` lives in `sdk/sdk.py`**, not in a shared types module. The runner logic is orchestration (train + evaluate + write), which belongs in the SDK. Confirmed.

7. **No `min_delta` parameter** — early stopping fires on any strict improvement. Confirmed: keep for v1.00.

---

## 17. References

- `HOMEWORK_BRIEF.md` § 6 (loss function), § 9 (open hyperparameters), § 10 (grading priorities).
- `docs/PRD.md` v1.01 § 4 (NFR-1 reproducibility, NFR-2 determinism), § 7 (PyTorch).
- `docs/PLAN.md` v1.01 § 6.1 (SDK contract), § 7 (training step sequence diagram), § 8 (LOC budget), § 9 (config schema), § 11.1 (seeding), § 11.4 (num_workers = 0).
- `docs/PRD_dataset_construction.md` v1.01 § 7 (tensor / dtype contract), § 13 item 1 (DataLoader shuffling per ADR-017).
- `docs/PRD_models.md` v1.01 § 13 item 4 (evaluate() may accept SignalExtractor internally), § 3.4 (model() invocation protocol).
- `docs/adr/ADR-014-results-layout.md` (M0, written) — run directory convention.
- `docs/adr/ADR-007-seeds-per-cell.md` (deferred) — number of seeds for the multi-seed grid.
- `docs/adr/ADR-013-config-override-policy.md` (deferred) — override path semantics.

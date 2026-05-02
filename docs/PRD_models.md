# PRD — Models (FC / RNN / LSTM)

> **Document version:** 1.01
> **Status:** Approved.
> **Owns:** the three neural architectures (FC, vanilla RNN, LSTM), the shared `SignalExtractor` base class, the selector-broadcast input reshape, the model registry, and the test plan for `services/models/`.
> **Companion:** `HOMEWORK_BRIEF.md` § 5 / § 7, `docs/PRD.md` v1.01, `docs/PLAN.md` v1.02, `docs/PRD_dataset_construction.md` v1.01, `docs/adr/ADR-002-pytorch.md` (M0, written), `docs/adr/ADR-003-selector-broadcast.md` (M0, written alongside this PRD), `docs/adr/ADR-009-component-decomposition.md` (M0, written).
>
> **Changelog from v1.00:** Arithmetic fix in § 5.3 LSTM parameter count (17 920 → 18 176; total 18 570 → 18 826). § 5.2 Consequences reframed — "RNN should excel" replaced with per-regime comparative claim. Seq-to-vector rationale neutralized in § 5.2 (replaces "stresses gradient flow" framing). Parameter-asymmetry disclosure + `EXP-008` ablation added to § 5.3 Consequences. `T-MD-18` assertion reworded to pin intent, not exact value. `predict()` alias removed from `SignalExtractor` — idiomatic call is `model(selector, w_noisy)`. Open questions Q1–Q4 resolved and recorded in § 13.

This is the dedicated PRD for the **models** subsystem. It defines exactly what `services/models/` must produce: three architectures behind one uniform `SignalExtractor` interface, the selector-broadcast reshape applied identically to RNN and LSTM, the module-level layout that fits the LOC budget, and the test plan that catches the architectural failure modes specific to this assignment (selector ignored, gross init bugs, head wired to wrong dimension).

Out of scope: signal generation (`PRD_signal_generation.md`), dataset construction (`PRD_dataset_construction.md`), training loop and evaluation metrics (`PRD_training_evaluation.md`).

---

## 1. Purpose

Given a `WindowExample` `(selector, w_noisy, w_clean)` from `services.dataset` (`PRD_dataset_construction.md` § 3.3), the models subsystem provides three `nn.Module` subclasses that:

- Accept the **canonical raw form** `(selector: (B,4), w_noisy: (B,10))` and return `w_pred: (B,10)`.
- Apply the selector-broadcast reshape internally (Dataset stays model-agnostic — `PRD_dataset_construction.md` § 7.3).
- Expose an identical external interface so that the training loop, the SDK, and the test harness can swap architectures with a single string flag (`"fc"` / `"rnn"` / `"lstm"`).
- Use PyTorch's default initialization in v1.00 (see § 6) so that the lecturer's RNN-vs-LSTM thesis is not accidentally short-circuited by clever weight scaling.

The deliverable is **fairness of comparison**, not raw accuracy. Every architectural choice in this PRD is justified relative to "does it preserve the apples-to-apples regime that lets the comparative study mean something?"

---

## 2. The Selector-Broadcast Scheme (Recap & Mechanics)

### 2.1 What ADR-003 fixes

The selector `C ∈ {e_1, e_2, e_3, e_4}` (a one-hot 4-vector) must reach RNN and LSTM at every timestep, in a way that is **identical** between the two architectures and comparable to FC. The chosen scheme is **broadcast**: tile `C` along the time axis and concatenate it onto each window sample. Full reasoning in `ADR-003-selector-broadcast.md`.

### 2.2 Concrete shapes per architecture

Given `selector: (B, 4)` and `w_noisy: (B, 10)`:

**FC.** Concatenate to a flat vector.
```text
fc_input = concat([selector, w_noisy], dim=-1)   # shape (B, 14)
```

**RNN / LSTM.** Tile selector along time, concatenate per step.
```text
sel_tiled  = selector.unsqueeze(1).expand(B, 10, 4)            # (B, 10, 4)
w_unsq     = w_noisy.unsqueeze(-1)                              # (B, 10, 1)
seq_input  = concat([w_unsq, sel_tiled], dim=-1)                # (B, 10, 5)
```
At each timestep `t`, the per-step feature is `[w_noisy[t], C[0], C[1], C[2], C[3]]`. RNN and LSTM share this shape exactly.

### 2.3 Where the reshape lives

The two reshape operations live **once** in `services/models/base.py` as private utility functions `_to_fc_input(selector, w_noisy)` and `_to_seq_input(selector, w_noisy)`. Each concrete model calls the appropriate utility at the top of its `forward()`. Duplicating the reshape per file is forbidden by the OOP / DRY rules in `SOFTWARE_PROJECT_GUIDELINES.md` § 3.2.

---

## 3. Setup, Inputs, Outputs (Building Block format — guidelines § 15.1)

### 3.1 Setup parameters (from `config/setup.json` § `model`)

```jsonc
"model": {
  "fc":   { "hidden": [64, 64] },
  "rnn":  { "hidden": 64, "layers": 1 },
  "lstm": { "hidden": 64, "layers": 1 }
}
```

| Key | Type | Default | Constraint |
| --- | --- | --- | --- |
| `model.fc.hidden` | list[int] | `[64, 64]` | length ≥ 1; each element > 0. |
| `model.rnn.hidden` | int | `64` | > 0. |
| `model.rnn.layers` | int | `1` | ≥ 1. |
| `model.lstm.hidden` | int | `64` | > 0. |
| `model.lstm.layers` | int | `1` | ≥ 1. |

The defaults match `PLAN.md` v1.01 § 9. Activation is `ReLU` for FC (locked, see § 5.1); `tanh` for vanilla RNN (PyTorch default, locked, see § 5.2); LSTM uses its standard sigmoid/tanh gate stack (PyTorch default, see § 5.3). Dropout is **off** in v1.00 (see § 5 alternatives).

### 3.2 Inputs (at `forward()` time)

| Argument | Shape | Dtype | Source |
| --- | --- | --- | --- |
| `selector` | `(B, 4)` | `float32` | `WindowExample.selector` (`PRD_dataset_construction.md` § 7.2) |
| `w_noisy` | `(B, 10)` | `float32` | `WindowExample.w_noisy` |

### 3.3 Outputs

| Field | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `w_pred` | `(B, 10)` | `float32` | predicted clean window for the channel selected by `selector` |

The output is consumed by the training loop's MSE loss (`PRD_training_evaluation.md`) and by evaluation metrics (per-frequency MSE, qualitative reconstructions).

### 3.4 The shared abstract base

```python
class SignalExtractor(nn.Module, abc.ABC):
    """
    Input:  (selector: (B,4) float32, w_noisy: (B,10) float32)
    Output: w_pred (B,10) float32
    Setup:  subclasses configure their own internal layers.
    """
    @abc.abstractmethod
    def forward(self, selector: torch.Tensor, w_noisy: torch.Tensor) -> torch.Tensor:
        ...
```

`SignalExtractor` is the only type that crosses the `services.models` boundary. Concrete classes (`FCExtractor`, `RNNExtractor`, `LSTMExtractor`) are private to the package — outside callers go through `models.build(kind, config)` (§ 7). Models are invoked as `model(selector, w_noisy)` via PyTorch's standard `__call__` / `forward()` protocol — no additional alias method.

---

## 4. The Input-Reshape Utility

`services/models/base.py` defines:

```python
def _to_fc_input(selector: torch.Tensor, w_noisy: torch.Tensor) -> torch.Tensor:
    """
    Input:  selector (B,4), w_noisy (B,10).
    Output: (B, 14) float32 — flat selector ⊕ window.
    Setup:  pure function. No allocation beyond torch.cat.
    """
    return torch.cat([selector, w_noisy], dim=-1)


def _to_seq_input(selector: torch.Tensor, w_noisy: torch.Tensor) -> torch.Tensor:
    """
    Input:  selector (B,4), w_noisy (B,10).
    Output: (B, 10, 5) float32 — per step [w_noisy[t], C[0..3]].
    Setup:  pure function. Selector is tiled (broadcast view) along time.
    """
    sel_tiled = selector.unsqueeze(1).expand(-1, w_noisy.shape[-1], -1)
    w_unsq    = w_noisy.unsqueeze(-1)
    return torch.cat([w_unsq, sel_tiled], dim=-1)
```

Both are pure, allocation-light, and tested directly (`T-MD-01`, `T-MD-02`). Concrete models do **not** reimplement them.

---

## 5. Architecture Specifications

Each subsection below uses the MADR shape (Context / Decision / Alternatives / Consequences) for the architectural choices that are not free-floating hyperparameters.

### 5.1 FC — `FCExtractor`

**Default architecture (v1.00):**
```text
input(14) → Linear(14, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, 10) → output(10)
```

Approximate parameter count: `(14·64+64) + (64·64+64) + (64·10+10) = 5 770`.

**Context.** FC is the non-temporal baseline. It has no notion of sequence and treats the 14-dim input as a static feature vector. Its job is to demonstrate what the task looks like *without* recurrence — a floor for the comparative study.

**Decision.** Two hidden layers of width 64 with ReLU. Final linear maps to 10. No dropout, no batch norm, no skip connections.

**Alternatives.**
- *Wider (e.g. 128 / 128).* Rejected for v1.00; ~5 700 params is already comfortably above the 14-dim input rank, and a wider FC would be over-parameterized for a smoke baseline.
- *Deeper (3+ layers).* Rejected; depth without skip connections invites optimization friction that isn't relevant to the thesis.
- *ReLU vs. GELU vs. tanh.* ReLU chosen because it is the dominant default in modern feed-forward MLPs and removes the saturation concern that would otherwise slow training of a small network.
- *Adding dropout (0.1 / 0.2).* Rejected for v1.00; the dataset is large relative to the parameter count (30 000 examples, ~5 700 params), so overfitting is unlikely. Reopen if `EXP-001` shows a generalization gap.

**Consequences.** A small, fast baseline that converges in a few epochs on CPU. Its MSE provides the reference floor; if RNN or LSTM cannot beat it on at least one frequency regime, the comparative study has nothing interesting to say.

### 5.2 RNN — `RNNExtractor`

**Default architecture (v1.00):**
```text
seq_input(B, 10, 5) → nn.RNN(input_size=5, hidden_size=64, num_layers=1,
                             nonlinearity='tanh', batch_first=True)
                    → take last timestep output[:, -1, :]   (B, 64)
                    → Linear(64, 10) → w_pred (B, 10)
```

Approximate parameter count: `nn.RNN(5,64,1)` = `5·64 + 64·64 + 2·64 = 4 544`; head `Linear(64,10)` = 650; total ≈ **5 194**.

**Context.** Vanilla RNN is the "short-memory" model in the lecturer's thesis: `tanh` activation + matrix multiplication of the hidden state at every step, prone to vanishing/exploding gradients on long contexts. The brief's window length (10) is deliberately short — the regime where RNN is *not* yet at its breaking point — but the architecture must still behave like a vanilla RNN, not be silently stabilized into something LSTM-like.

**Decision.** Single-layer `nn.RNN` with `tanh` nonlinearity, hidden size 64. Output head consumes the **final-timestep hidden output** (`output[:, -1, :]`) and projects to 10 dimensions via a single linear layer (sequence-to-vector pattern). PyTorch default init (uniform `[-1/√hidden, 1/√hidden]`) — see § 6.

**Alternatives.**
- *Sequence-to-sequence head* (per-timestep projection, concatenate). Rejected for v1.00. The sequence-to-vector head requires the architecture to compress all 10 timesteps of information into a single hidden representation before projection — the regime where gating capacity (LSTM) and per-timestep recurrence (RNN) are most clearly distinguishable. A sequence-to-sequence head allows each timestep to read out independently, weakening the comparison. Listed as planned ablation `EXP-007` to verify the head choice does not bias conclusions.
- *ReLU instead of tanh.* Rejected. ReLU in vanilla RNNs partially circumvents vanishing gradients and would short-circuit the thesis. We *want* the vanilla RNN to behave like a textbook vanilla RNN (`tanh`, default init, no clever tricks).
- *Multi-layer (`num_layers ≥ 2`).* Rejected for v1.00; adds depth and parameters that aren't part of the comparison. Reopen if a future ablation needs it.
- *Bidirectional.* Unidirectional in v1.00 — see § 13. Bidirectional RNN → planned ablation `EXP-005`.

**Consequences.** The RNN should perform competitively with FC on high-frequency components (`s_4` at 200 Hz — two full cycles per window, where local within-window structure is sufficient) and struggle on low-frequency components (`s_1` at 2 Hz — period 500 samples, where the entire window represents a near-linear slope that requires retaining information across the full sequence). The thesis is that LSTM, not RNN, gains advantage as the period grows — the architecture is set up so that a confirmation or refutation actually means something.

### 5.3 LSTM — `LSTMExtractor`

**Default architecture (v1.00):**
```text
seq_input(B, 10, 5) → nn.LSTM(input_size=5, hidden_size=64, num_layers=1,
                              batch_first=True)
                    → take last timestep output[:, -1, :]   (B, 64)
                    → Linear(64, 10) → w_pred (B, 10)
```

Approximate parameter count: `nn.LSTM(5,64,1)` = `4·(5·64 + 64·64 + 2·64) = 18 176`; head 650; total ≈ **18 826**. (LSTM has 4× the gate count vs. RNN.)

**Context.** LSTM is the "long-memory" model in the lecturer's thesis. Its forget / input / output gates are designed to preserve gradient flow over long sequences. On the assignment's 10-step windows, the gating advantage shows up most clearly on **low-frequency components** where the entire window is needed.

**Decision.** Single-layer `nn.LSTM`, hidden size 64 (matching RNN for fairness). Same sequence-to-vector head as RNN. PyTorch default init for both `i_h` / `h_h` weight blocks and forget-gate bias (= 0 by default). See § 6 for the explicit decision *not* to set forget-gate bias to 1.0.

**Alternatives.**
- *Hidden size larger than RNN's* (e.g., 128). Rejected — would make LSTM "win" on parameter count, polluting the comparison. Same hidden size keeps the comparison about gating, not capacity.
- *Forget-gate bias = 1.0* (the Jozefowicz / Gers heuristic). Rejected for v1.00; setting it to 1.0 systematically pushes LSTM toward the "remember everything" regime, which would bias the thesis evaluation. PyTorch default (bias = 0) is the honest baseline.
- *Multi-layer / bidirectional.* Same disposition as RNN — single-layer locked, bidirectional surfaced in § 13.

**Consequences.** LSTM should beat RNN on low-frequency components and roughly tie or modestly beat on mid frequencies. If LSTM wins everywhere by a wide margin, the most likely cause is parameter count rather than gating — a result worth diagnosing in the analysis notebook.

Note that LSTM at hidden=64 has ~18 826 parameters versus RNN's ~5 194 — a ~3.6× ratio driven by the four gate matrices in LSTM. The comparison at equal hidden size is therefore about gating AND parameters jointly. A parameter-matched comparison (RNN at hidden ≈ 128, yielding ~18 570 params) is listed as planned ablation `EXP-008`. The equal-hidden choice in v1.00 follows the lecturer's framing of the thesis (architectural properties at comparable per-step compute) rather than equal-capacity matching.

---

## 6. Initialization Strategy

**Decision: PyTorch defaults for all three models in v1.00.** No custom initialization, no Xavier/Glorot, no orthogonal init for recurrent weights, no forget-gate bias = 1.

**Rationale.** RNN and LSTM are sensitive to initialization. PyTorch's defaults are:
- `nn.Linear`: weights `~ U(-√(1/fan_in), √(1/fan_in))`, bias `~ U(-√(1/fan_in), √(1/fan_in))`.
- `nn.RNN` / `nn.LSTM`: all weight matrices `~ U(-√(1/hidden), √(1/hidden))`, biases initialized to zero.

These defaults are "just barely" stable enough to train on short sequences without catastrophic gradient explosion or full vanishing. Replacing them with cleverer schemes (e.g., orthogonal init for `weight_hh`, identity-init for the recurrent matrix, forget-gate bias = 1) would systematically *fix* the gradient pathologies that the lecturer's RNN-vs-LSTM thesis is built on. Specifically:

- *Orthogonal `weight_hh` init* mitigates RNN vanishing gradients. The RNN would behave less like a textbook vanilla RNN and more like a partial LSTM, blunting the comparison.
- *Forget-gate bias = 1.0 (LSTM)* makes the LSTM strongly biased toward retention, which would inflate its low-frequency advantage even on tasks where gating isn't the deciding factor.

We want each architecture to behave like its textbook self. Defaults achieve that. If `EXP-001` reveals catastrophic instability (e.g., NaN losses, RNN failing to make any progress at all), initialization is reopened and `ADR-019-init-strategy.md` is opened then. Until that trigger, defaults are locked.

---

## 7. Public API & Model Registry

`services/models/` exposes:

```python
ModelKind = Literal["fc", "rnn", "lstm"]

@dataclass(frozen=True)
class ModelConfig:
    fc:   FCConfig
    rnn:  RNNConfig
    lstm: LSTMConfig

@dataclass(frozen=True)
class FCConfig:    hidden: tuple[int, ...]   # default (64, 64)
@dataclass(frozen=True)
class RNNConfig:   hidden: int; layers: int  # defaults 64, 1
@dataclass(frozen=True)
class LSTMConfig:  hidden: int; layers: int  # defaults 64, 1

def build(kind: ModelKind, config: ModelConfig) -> SignalExtractor:
    """
    Input:  kind ∈ {"fc","rnn","lstm"}, ModelConfig.
    Output: a SignalExtractor instance with weights at PyTorch defaults.
    Setup:  registry dispatch — looks up the concrete class by kind.
            Raises ValueError on unknown kind.
    """
```

`SignalExtractor` is the only public type from this package. Concrete classes (`FCExtractor`, `RNNExtractor`, `LSTMExtractor`) are private to the package — outside callers go through `models.build(kind, config)` exclusively. Models are invoked as `model(selector, w_noisy)` — PyTorch's standard `__call__` / `forward()` protocol.

The registry is a simple `dict[ModelKind, type[SignalExtractor]]` populated in `services/models/__init__.py`.

---

## 8. Module Layout & LOC Budget

`PLAN.md` v1.01 § 8 allocates the following ceilings for `services/models/`:

| File | PLAN.md ceiling | v1.00 internal target |
| --- | --- | --- |
| `__init__.py` | 10 | ~15 (registry + public exports) |
| `base.py` | 60 | ~40 (`SignalExtractor`, `_to_fc_input`, `_to_seq_input`) |
| `fc.py` | 80 | ~30 (`FCExtractor` + tiny config dataclass references) |
| `rnn.py` | 80 | ~30 |
| `lstm.py` | 80 | ~30 |
| **Total** | **310** | **~145** |

> **Note on the budget.** The "internal target" column is a tighter goal than what `PLAN.md` ratifies; the contractual ceiling (per-file, per `SOFTWARE_PROJECT_GUIDELINES.md` § 2.2) is the PLAN.md column. We aim for the tighter total (~145 LOC) but no test fails because we use 200. The PLAN.md v1.02 sweep should consider reducing the model package allocation to reflect that PyTorch handles most of the heavy lifting; tracked in the v1.02 follow-up checklist in `PRD_dataset_construction.md` § 15.

If any single file overflows its ceiling: split per `SOFTWARE_PROJECT_GUIDELINES.md` § 2.2 (extract a mixin, split read/write halves, etc.) — never compress to fit.

---

## 9. Test Specification

All tests live in `tests/unit/test_models/`, mirroring the package structure (`test_base.py`, `test_fc.py`, `test_rnn.py`, `test_lstm.py`, plus a cross-cutting `test_registry.py`). Tests are written before implementation per TDD.

### 9.1 Unit tests

| ID | Behavior | Assertion |
| --- | --- | --- |
| T-MD-01 | `_to_fc_input` shape & dtype | input `(B,4) + (B,10)` → output `(B, 14)`, `float32`; values match `concat([selector, w_noisy], dim=-1)`. |
| T-MD-02 | `_to_seq_input` shape & dtype | input `(B,4) + (B,10)` → output `(B, 10, 5)`, `float32`; per-timestep `t`, last 4 dims equal `selector` and first dim equals `w_noisy[..., t]`. |
| T-MD-03 | Registry dispatch | `build("fc", cfg)` returns `FCExtractor`; same for `"rnn"` / `"lstm"`. Unknown kind → `ValueError`. |
| T-MD-04 | Forward shape — FC | for any batch size `B`, `FCExtractor(...)(sel, win).shape == (B, 10)`, `dtype == float32`. |
| T-MD-05 | Forward shape — RNN | same as T-MD-04 for `RNNExtractor`. |
| T-MD-06 | Forward shape — LSTM | same as T-MD-04 for `LSTMExtractor`. |
| T-MD-07 | Selector responds — FC | with fixed `w_noisy`, swapping `selector` from `e_1` to `e_3` produces a different output (L∞ distance > 1e-6 at random init). Catches the "model ignores `C`" failure mode. |
| T-MD-08 | Selector responds — RNN | same assertion for `RNNExtractor`. |
| T-MD-09 | Selector responds — LSTM | same assertion for `LSTMExtractor`. |
| T-MD-10 | Trainability smoke — FC | overfit a fixed 4-example mini-dataset with full-batch SGD (lr=1e-2) for 200 steps; final MSE < 1e-3. Catches gross architectural bugs (wrong output dim, dead activation, exploding init). |
| T-MD-11 | Trainability smoke — RNN | same procedure; final MSE < 5e-3 (looser bound; vanilla RNN converges slower than FC even on a 4-example overfit). |
| T-MD-12 | Trainability smoke — LSTM | same as T-MD-11. |
| T-MD-13 | Parameter count — FC | `sum(p.numel())` ∈ `[5_500, 6_000]` with default config. Catches accidental width / depth changes. |
| T-MD-14 | Parameter count — RNN | `sum(p.numel())` ∈ `[5_000, 5_500]`. |
| T-MD-15 | Parameter count — LSTM | `sum(p.numel())` ∈ `[18_200, 18_900]`. |
| T-MD-16 | Determinism | with `torch.manual_seed(s)` set, two builds with the same `(kind, config)` produce models whose weights are bit-identical. |
| T-MD-17 | RNN nonlinearity is `tanh` | inspect `RNNExtractor.rnn.nonlinearity == "tanh"`. Pinned because § 5.2 forbids the silent ReLU swap. |
| T-MD-18 | LSTM forget-gate bias defaults | for the LSTM, `\|bias_ih_l0[hidden:2*hidden]\| < 0.5` — confirming we are NOT applying the Jozefowicz heuristic of forget-bias = 1.0. If a future PyTorch version changes this default, the test fires and `ADR-019-init-strategy` is reopened. Test file references § 6 of this PRD. |
| T-MD-19 | Gradient flow on a single step | for each model, a forward + `loss.backward()` produces non-zero `.grad` on every parameter tensor. Catches detached subgraphs and dead branches. |
| T-MD-20 | Output is differentiable in `selector` | `selector.requires_grad_(True)`; after backward on the output sum, `selector.grad` is non-zero. Strengthens T-MD-07/08/09: not only does the output depend on `selector`, the dependency is differentiable. |

### 9.2 Coverage target

`services/models/` is targeted at ≥ 90% line coverage. The `forward()` paths are short and fully exercised by the trainability and shape tests; the registry and reshape utilities are pure functions trivial to cover.

---

## 10. Edge Cases

| Case | Behavior |
| --- | --- |
| `selector` not one-hot (e.g., all zeros, multiple ones) | Model still runs (the architecture has no enforcement). The `Dataset` guarantees one-hot input (`T-DS-09`); models do not re-validate. Documented but not tested at the model layer. |
| `B = 1` | Forward must work. Tested implicitly via T-MD-04..06 with both `B=1` and `B=256`. |
| `w_noisy` length ≠ 10 | Model accepts it for FC (linear input dim is locked at 14, so this raises `RuntimeError` from `nn.Linear`); for RNN/LSTM, it accepts any sequence length (the layers are length-agnostic) but `_to_seq_input` produces `(B, T, 5)`. The 10-sample contract is enforced at the Dataset boundary, not the model boundary. |
| `selector.shape[-1] ≠ 4` | Same — enforced upstream at the Dataset boundary. The model would technically accept any selector dim (the FC `Linear(14, ...)` would fail; RNN/LSTM `_to_seq_input` would still produce a `(B, T, 1+sel_dim)` tensor, and `nn.RNN(input_size=5)` would fail). |
| Unknown `kind` in `build()` | `ValueError("unknown model kind: {kind}")`. Tested by T-MD-03. |
| `config.fc.hidden = []` | `ValueError` at `FCExtractor.__init__`. The minimum is one hidden layer. |
| `config.rnn.layers > 1` | Allowed; PyTorch handles multi-layer RNN. The default config (M0) sticks to 1; multi-layer is exercised only if a future ablation enables it. |
| CPU-only environment | All three models train on CPU. Device placement is the training loop's concern, not the model's. |

---

## 11. Acceptance Criteria

The models subsystem ships when:

- AC-MD-1: All tests `T-MD-01` through `T-MD-20` pass.
- AC-MD-2: Each file in `services/models/` is at or below its `PLAN.md` § 8 per-file ceiling. Aggregate target ~145 LOC; aggregate ceiling 310 LOC.
- AC-MD-3: Coverage on `services/models/` is ≥ 90%.
- AC-MD-4: `ruff check` clean.
- AC-MD-5: Building Block docstrings on `SignalExtractor`, `FCExtractor`, `RNNExtractor`, `LSTMExtractor`, all `*Config` dataclasses, `build`, and the two reshape utilities.
- AC-MD-6: An integration test `tests/integration/test_models_smoke.py` runs `build("fc"|"rnn"|"lstm", default_config)` → `model(selector, w_noisy)` for one mini-batch from a real `WindowDataset` (driven by the SDK), and asserts the output shape and finite-loss invariant.
- AC-MD-7: A short notebook cell shows the same noisy window decoded by all three trained-default models for the same selector — a qualitative sanity figure for the analysis chapter. Figure committed to `assets/figures/models_qualitative.png` (delivered in M5, not M3+M4).

---

## 12. Alternatives Considered (high-level — full text in ADRs / per-section)

- **Hidden-state initialization for RNN/LSTM** instead of selector-broadcast. Rejected (`ADR-003`); see § 2.1 for the comparability argument and § 13 for the post-v1.00 ablation plan.
- **Prefix-step injection** of the selector. Rejected (`ADR-003`); changes effective context length.
- **Sequence-to-sequence head** instead of sequence-to-vector. Rejected for v1.00 (§ 5.2); listed as `EXP-007` ablation.
- **ReLU in vanilla RNN** instead of `tanh`. Rejected (§ 5.2); short-circuits the vanishing-gradient regime that is part of the comparison.
- **Forget-gate bias = 1.0 in LSTM** (Jozefowicz heuristic). Rejected (§ 5.3 / § 6); biases LSTM toward retention and pollutes the thesis evaluation.
- **Larger LSTM hidden size** to "match performance." Rejected (§ 5.3); would make any LSTM win attributable to capacity rather than gating. Parameter-matched comparison (RNN hidden ≈ 128) deferred to `EXP-008`.
- **Custom Xavier / orthogonal init**. Rejected (§ 6); we want vanilla architectures to misbehave the way vanilla architectures misbehave.
- **Dropout in v1.00**. Rejected (§ 5.1); the dataset:parameter ratio doesn't suggest overfitting risk. Reopen if `EXP-001` shows a generalization gap.
- **Exposing concrete model classes across the package boundary.** Rejected; `SignalExtractor` is the only public type. Outside callers go through `build()`.

---

## 13. Decisions Locked at v1.01 Review

The four open questions raised in v1.00 have been resolved:

1. **Bidirectional RNN/LSTM.** Unidirectional in v1.00. The lecturer's thesis is framed around a left-to-right pass (gradient pathologies accumulate from early to late timesteps); bidirectional would partially obscure that mechanism. Bidirectional RNN/LSTM → planned ablation `EXP-005` (post-v1.00). No new ADR required — decision is local to model architecture and does not affect the SDK contract or component boundaries.
2. **Sequence-to-vector vs. sequence-to-sequence head.** Sequence-to-vector locked in v1.00; see revised rationale in § 5.2 / § 5.3. Sequence-to-sequence → planned ablation `EXP-007`.
3. **Hidden-state initialization as a v1.00 ablation.** Deferred. The ablation stays as `EXP-006` (post-v1.00); it is not added to the v1.00 package. Adding it now would inflate the package and obscure the primary three-way comparison.
4. **`evaluate()` accepting `nn.Module` internally.** Resolved: internal service-to-service calls (e.g. `services.training` → `services.evaluation`) may pass a `SignalExtractor` instance directly. The restriction on raw `nn.Module` applies only to the SDK's public surface. This is documented in `PRD_training_evaluation.md` (to be written).

---

## 14. References

- `HOMEWORK_BRIEF.md` § 5 (model contract), § 7 (mandatory architectures), § 10 (grading priorities).
- `docs/PRD.md` v1.01 § 4 (NFR-3 uniform interface), § 7 (modeling library = PyTorch).
- `docs/PLAN.md` v1.01 § 6.1 (SDK contract), § 7.2 (per-batch shapes), § 8 (LOC budget), § 13 (ADR index).
- `docs/PRD_dataset_construction.md` v1.01 § 7 (tensor / dtype contract delivered to models).
- `docs/PRD_training_evaluation.md` (to be written; loss = MSE, optimizer = Adam by default).
- `docs/adr/ADR-002-pytorch.md` (M0, written).
- `docs/adr/ADR-003-selector-broadcast.md` (M0, written alongside this PRD).
- `docs/adr/ADR-009-component-decomposition.md` (M0, written).

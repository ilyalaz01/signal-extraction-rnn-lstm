# PRD — Dataset Construction

> **Document version:** 1.01
> **Status:** Approved.
> **Owns:** how a `Corpus` (from `signal_gen`) becomes train / val / test PyTorch datasets. This is the bridge between signal generation and model training.
> **Companion:** `HOMEWORK_BRIEF.md` § 4, `docs/PRD.md` v1.01, `docs/PLAN.md` v1.01, `docs/PRD_signal_generation.md` v1.01, `docs/adr/ADR-004-dataset-size.md`, `docs/adr/ADR-016-random-sampling-stationary.md`, `docs/adr/ADR-017-dataloader-shuffling.md`.
>
> **Changelog from v1.00:** § 5 rewritten — disjoint `t_0` ranges replaced with iid random sampling on a stationary process (ADR-016). Setup parameters trimmed: `split_strategy`, `train_fraction`, `val_fraction` removed. Public function `compute_split_ranges` removed. Tests `T-DS-01..T-DS-03` dropped (function gone); `T-DS-08` typo fixed; `T-DS-11` replaced by `T-DS-11'` (KS equivalence between split distributions); `T-DS-13` updated to the full pool; `T-DS-18` (meta idempotency) added. Open questions Q1, Q2 locked (ADR-017 and the `meta()` method); Q3 is conditional on `EXP-001` and `ADR-018` is no longer pre-planned. New `AC-DS-8` (meta idempotency) and `AC-DS-9` (t_0-histogram figure).

This is the dedicated PRD for the **dataset construction** subsystem. It defines the window-sampling procedure, the train/val/test split, the tensor contract delivered to models, and the test plan for `services/dataset.py`.

Out of scope here: signal generation (lives in `PRD_signal_generation.md`), model definitions (in `PRD_models.md`), training loop (in `PRD_training_evaluation.md`).

---

## 1. Purpose

Given a `Corpus` produced by `services.signal_gen`, produce three PyTorch `Dataset`s — `train`, `val`, `test` — such that:

- Each example is a `(C, W_noisy, W_clean)` triple matching the homework brief's contract (§ 4 of HOMEWORK_BRIEF).
- Each split draws `t_0` iid uniformly from the full pool `[0, N - W]`; splits differ only in example count and RNG seed (see § 5 for the stationarity rationale, locked in `ADR-016`).
- The example count matches the locked sizes from `ADR-004` (30 000 / 3 750 / 3 750).
- Construction is fully deterministic given `(corpus, seed)`.

The output of this subsystem is a `SplitDatasets` dataclass containing the three datasets plus provenance.

---

## 2. The Window Contract (recap of HOMEWORK_BRIEF § 4)

A single example is constructed as:

1. Draw a one-hot selector $C \in \{e_1, e_2, e_3, e_4\}$ uniformly.
2. Draw a start index $t_0 \in [0, N - W]$ uniformly **from the full pool**. The same pool is used for all three splits (see § 5).
3. Cut $W_{\text{noisy}} = \tilde{S}[t_0 : t_0 + W]$ from `corpus.noisy_sum`. Length $W = 10$.
4. Cut $W_{\text{clean}} = s_k[t_0 : t_0 + W]$ from `corpus.clean[k]`, where $k = \arg\max(C)$.
5. Emit the triple $(C, W_{\text{noisy}}, W_{\text{clean}})$.

**Important:** the target is cut from the *clean* per-sinusoid signal, **not** from `clean_sum`. The whole point of the task is to extract one component out of the noisy mixture.

The Dataset returns this **canonical raw form** in every case. Model-specific reshaping (FC's flat 14-vector, RNN/LSTM's `(10, 5)` per-step format) is the model's responsibility, not the Dataset's. This keeps the Dataset reusable across all three architectures.

---

## 3. Setup, Inputs, Outputs (Building Block format — guidelines § 15.1)

### 3.1 Setup parameters (from `config/setup.json` § `dataset`)

| Field | Type | Default | Constraint |
| --- | --- | --- | --- |
| `window` | int | `10` | == 10. Locked by HOMEWORK_BRIEF § 4.2. Anything else is a `ValueError`. |
| `n_train` | int | `30000` | > 0. ADR-004. |
| `n_val` | int | `3750` | > 0. ADR-004. |
| `n_test` | int | `3750` | > 0. ADR-004. |

There is no `split_strategy` field in v1.00: `ADR-016` fixes a single strategy (random iid sampling on a stationary process). A future ablation that introduces a second strategy will reintroduce the field at that time. Likewise no `train_fraction` / `val_fraction` — splits are defined by absolute counts, not fractions, since the underlying `t_0` pool is shared across splits and there is no range-slicing to perform.

### 3.2 Inputs

- A `Corpus` (frozen dataclass from `services.signal_gen`).
- A `DatasetConfig` (parsed from `config['dataset']`).
- A `seed: int` for the sampling RNG. (This is a **second** seed, distinct from the corpus seed; same seed → same draws.)

No filesystem I/O. No randomness beyond the supplied seed. The subsystem is purely functional.

### 3.3 Outputs

```python
@dataclass(frozen=True)
class WindowExample:
    """
    Input:  one __getitem__ call on a WindowDataset.
    Output: this triple.
    Setup:  immutable; tensors are torch.float32 (signals) / torch.float32 (selector — see § 7.2).
    """
    selector: torch.Tensor    # shape (4,), float32, exactly one entry == 1.0
    w_noisy:  torch.Tensor    # shape (10,), float32
    w_clean:  torch.Tensor    # shape (10,), float32

@dataclass(frozen=True)
class SplitDatasets:
    """
    Input:  produced by services.dataset.build_split_datasets(corpus, config, seed).
    Output: this object.
    Setup:  three Datasets + provenance fields.
    """
    train: WindowDataset
    val:   WindowDataset
    test:  WindowDataset
    config: DatasetConfig
    corpus_seed: int
    sampling_seed: int

class WindowDataset(torch.utils.data.Dataset):
    """
    Input:  index i in [0, len-1).
    Output: WindowExample at position i (deterministic — see § 4).
    Setup:
        Carries the corpus reference, the precomputed (t_0, channel_idx)
        index table for this split, the window size W=10, and a
        `split_name` ∈ {"train", "val", "test"} used by `meta()` (§ 8).
    """
```

`WindowDataset` does **not** carry a copy of the signals — it holds a reference to the same `Corpus` (read-only), so the three splits share the underlying ~400 KB of signal data. Only the `(t_0, channel_idx)` index tables and the `split_name` label are split-specific.

---

## 4. Sampling Procedure

The class **precomputes** an index table per split. Each row is `(t_0, k)` where `k` is the index of the `1` in $C$ (so we store the integer instead of the full one-hot to save memory). The selector tensor is materialized lazily inside `__getitem__`.

```text
For split S with size n_S:
  rng = np.random.default_rng(seed_S)              # split-specific seed (§ 6)
  t0_pool = arange(0, N - W + 1)                    # full pool [0, 9990]; shared across splits (§ 5)
  t0      = rng.choice(t0_pool, size=n_S, replace=True)
  k       = rng.integers(0, 4, size=n_S)
  index_table_S = np.stack([t0, k], axis=1)         # shape (n_S, 2), int64
```

The same `t0_pool` feeds all three splits — the only thing that distinguishes train from val from test is the per-split RNG seed and the size `n_S`. This is what makes the splits iid samples from the same window distribution; the split labels exist for reporting and reproducibility, not to define different distributions (`ADR-016`).

Sampling is **with replacement** so that train, val, and test draws are statistically iid from the same `Uniform({0, ..., N-W}) × Uniform({0,1,2,3})` distribution. Without-replacement would slightly distort the distribution (especially in train, where 30 000 draws come from a 39 964-slot `(t_0, k)` pool) and break the iid property that `ADR-016` relies on.

Each `__getitem__(i)` does:

```text
t0, k = index_table[i]
w_noisy = torch.from_numpy(corpus.noisy_sum[t0 : t0 + W]).float()
w_clean = torch.from_numpy(corpus.clean[k, t0 : t0 + W]).float()
selector = one_hot(k, num_classes=4).float()
return WindowExample(selector, w_noisy, w_clean)
```

`__getitem__` is a pure function of `(i, corpus, index_table)`. Same `i` → same example, every time. This is what makes deterministic training and reproducible test metrics work.

---

## 5. Split Strategy: Random Sampling on a Stationary Process

This decision is locked in `ADR-016-random-sampling-stationary.md` (M0). Summary below.

The corpus has `N = 10 000` samples. With `W = 10`, there are `N - W + 1 = 9 991` valid start indices `t_0 ∈ [0, 9990]`. Combined with the 4-class selector, the full sampling pool has `9 991 × 4 = 39 964` unique `(t_0, k)` slots.

**All three splits draw from the same pool.** Each split samples `t_0` iid uniformly from `[0, 9 990]` and `k` iid uniformly from `{0, 1, 2, 3}`. The splits differ only in:

1. **Example count** — `n_train = 30 000`, `n_val = 3 750`, `n_test = 3 750`.
2. **RNG seed** — each split receives a distinct child of `sampling_seed` via `SeedSequence.spawn(3)`, so draws are statistically independent across splits (§ 6).

**Why no disjoint t_0 ranges.** The signal generation process is **stationary** — every `t ∈ [0, N)` is governed by the same generative law (sum of four fixed-frequency, fixed-amplitude sinusoids with stationary i.i.d. amplitude and phase noise; `PRD_signal_generation.md` § 3). The marginal distribution over windows is therefore identical for every `t_0`. Partitioning the `t_0` axis into disjoint ranges across splits would not measure a generalization gap — the train and test distributions would be mathematically identical regardless of the partition. A "disjoint split" here would be cosmetic, not a generalization probe. `ADR-016` records the alternatives considered (true disjoint with `W-1` buffer; "shoulder-disjoint" — the v1.00 draft, caught in review; per-split corpora) and why each was rejected.

> **Framing sentence (reproduced verbatim in `ADR-016` and required verbatim in `README.md` § Methodology when M6 ships):**
> Splits exist by example count (30 000 / 3 750 / 3 750) for reproducibility and reporting, but `t_0` is drawn iid from the full pool `[0, N - W]` in all three splits because the underlying signal process is stationary; disjoint `t_0` ranges would not measure a generalization gap.

**Visual proof.** The analysis notebook delivered at M5 must include a figure overlaying the empirical `t_0` histograms of the three splits — they should be statistically indistinguishable. This is `AC-DS-9`.

---

## 6. Random Number Generators — Three Distinct Seeds

The project uses three independent seed slots:

| Slot | Source | Consumed by | Documented in |
| --- | --- | --- | --- |
| `runtime.seed` | `config['runtime']['seed']` | global `shared.seeding` (python/numpy/torch global RNGs) | `PLAN.md` § 11.1 |
| `corpus_seed` | derived: `seeding.derive(runtime.seed, "corpus")` | `signal_gen.generate_corpus` | `PRD_signal_generation.md` § 5.3 |
| `sampling_seed` | derived: `seeding.derive(runtime.seed, "sampling")` | `dataset.build_split_datasets` | this PRD |

The derivation is `np.random.SeedSequence(runtime_seed).spawn(2)` so a single user-supplied seed deterministically yields both a corpus seed and a sampling seed. This ensures *one* knob controls *all* randomness in data generation. Inside `build_split_datasets`, the `sampling_seed` is further spawned into three split-specific seeds for train/val/test — that way you can change `n_test` without changing the train order.

---

## 7. Tensor & Dtype Conventions

### 7.1 Signal tensors

All signal-derived tensors (`w_noisy`, `w_clean`, model inputs, model outputs) are `torch.float32`. The corpus arrays are already `float32` per `PRD_signal_generation.md` § 3.3, so the conversion in `__getitem__` is a no-cost view.

### 7.2 Selector tensor

The selector $C$ is also `torch.float32`. It is **not** `torch.long` or `torch.bool`, because it is concatenated/broadcast with signal samples that are floats; mixing dtypes here would force casts inside the model. Storing the integer label `k` separately (as `torch.long`) is left as an opt-in for future losses that need the class index — not used in v1.00.

### 7.3 Batched shapes (delivered by `DataLoader` collation)

After default collation with batch size `B`:

| Tensor | Shape | Notes |
| --- | --- | --- |
| `selector` | `(B, 4)` | one-hot, float32 |
| `w_noisy` | `(B, 10)` | float32 |
| `w_clean` | `(B, 10)` | float32, the training target |

The model's `forward` is responsible for reshaping `(selector, w_noisy)` into either a flat 14-vector (FC) or a `(B, 10, 5)` sequence (RNN/LSTM). The Dataset stays model-agnostic.

### 7.4 Device placement

Datasets stay on CPU. The training loop moves per-batch tensors to the configured device. Pinned memory is **not** used in v1.00 because `num_workers=0` (PLAN.md § 11.4).

---

## 8. Public API

`services/dataset.py` exposes:

```python
@dataclass(frozen=True)
class DatasetConfig:
    window: int
    n_train: int
    n_val: int
    n_test: int

class WindowDataset(torch.utils.data.Dataset):  # defined in § 3.3
    """
    In addition to the standard Dataset protocol, exposes:
        meta(i: int) -> dict
    """
    def meta(self, i: int) -> dict:
        """
        Input:  index i in [0, len-1).
        Output: {"t_0": int, "k": int, "split_name": str} for plotting/debugging.
        Setup:  pure function of (i, index_table, split_name); idempotent
                across repeated calls for the same i (verified by T-DS-18).
                Does not affect, and is not affected by, __getitem__.
        """

@dataclass(frozen=True)
class SplitDatasets:  # defined in § 3.3
    ...

def build_split_datasets(
    corpus: Corpus, config: DatasetConfig, sampling_seed: int
) -> SplitDatasets:
    """
    Input:  Corpus, DatasetConfig, an int seed.
    Output: SplitDatasets with three populated WindowDatasets.
    Setup:  uses np.random.SeedSequence(seed).spawn(3) for per-split seeds.
            All three datasets share the corpus by reference (no copy)
            and sample t_0 from the same full pool [0, N - W] (§ 5).
    """
```

Anything else is private. The `compute_split_ranges` function from v1.00 is removed — there are no per-split ranges to compute. Total LOC budget for `dataset.py` is ~100 (down from ~120 in v1.00; PLAN.md § 8 will be updated in the v1.02 PLAN sweep).

---

## 9. Test Specification

All tests live in `tests/unit/test_dataset.py`, written before implementation per TDD.

### 9.1 Unit tests

| ID | Behavior | Assertion |
| --- | --- | --- |
| T-DS-04 | `window == 10` enforced | other values raise `ValueError`. |
| T-DS-05 | Dataset length | `len(ds.train) == n_train`; same for val/test. |
| T-DS-06 | Determinism | same `(corpus, config, seed)` → identical example tensors at every index across two builds. |
| T-DS-07 | Index table dtype | the precomputed `(t_0, k)` table is `int64`, shape `(n, 2)`. |
| T-DS-08 | `__getitem__` shapes | `selector.shape == (4,)`, `w_noisy.shape == w_clean.shape == (10,)`, all `float32`. |
| T-DS-09 | One-hot validity | `selector.sum() == 1.0` and exactly one entry == 1.0. |
| T-DS-10 | Selector ↔ target consistency | for the example at index `i`, the integer `k = selector.argmax()` matches the channel of `w_clean` (verified by checking that `w_clean` equals `corpus.clean[k, t0:t0+W]` for the recovered `t0`). |
| T-DS-11' | Split distributions are statistically equivalent | two-sample KS test on `train.t_0` vs `test.t_0` does **not** reject at α=0.001. (Replaces the v1.00 "no leakage" test, which is meaningless under the random-sampling design — see `ADR-016`. This test asserts the *expected* equivalence as a positive sanity check on the shared-pool design.) |
| T-DS-12 | Class balance (statistical) | over 30 000 train examples, each `k ∈ {0,1,2,3}` appears `7500 ± 4σ` times where `σ = sqrt(30000 * 0.25 * 0.75)`. |
| T-DS-13 | `t_0` uniform within split (statistical) | KS test of `train.t_0` against `Uniform[0, 9990]` does not reject at α=0.001. (Pool is the full `[0, N - W]`, per § 5 / `ADR-016`.) |
| T-DS-14 | DataLoader integration | a default `DataLoader(ds.train, batch_size=256)` produces batched shapes `(256, 4) / (256, 10) / (256, 10)`. |
| T-DS-15 | Memory sharing | the three splits share `corpus.clean`, `corpus.noisy_sum` by reference (verified by `id(...)`). |
| T-DS-16 | Sampling-seed independence from corpus | same corpus + different sampling seeds → different `t_0` tables. |
| T-DS-17 | Re-creating with fewer test windows preserves train order | building with `n_test=100` then `n_test=200` (same other params) yields identical train tables. (Tests the spawn-based per-split seeding from § 6.) |
| T-DS-18 | `meta(i)` is idempotent | calling `ds.meta(i)` twice returns equal dicts; the dict has keys `{"t_0", "k", "split_name"}` with `t_0 ∈ [0, N - W]`, `k ∈ {0,1,2,3}`, `split_name` matching the split. Required by `AC-DS-8`. |

**Note on dropped tests.** v1.00 included `T-DS-01`, `T-DS-02`, `T-DS-03` covering `compute_split_ranges`. With the function gone (§ 8), those tests are dropped. Test IDs are not renumbered to preserve historical references. v1.00 `T-DS-11` (no leakage) is replaced by `T-DS-11'` because under the random-sampling design there is no disjointness to enforce — the natural successor test is a positive equivalence check.

### 9.2 Coverage target

`dataset.py` is targeted at ≥ 95% line coverage; the module is small and pure, like `signal_gen.py`.

---

## 10. Edge Cases

| Case | Behavior |
| --- | --- |
| `n_train + n_val + n_test == 0` | `ValueError` — at least one split must be non-empty. |
| `n_S > unique_(t_0,k)_slots` (for any split) | Cannot occur in the v1.00 configuration: the shared pool has 39 964 unique slots and the largest split is 30 000. If a future config inflates a split count past the pool size, sampling-with-replacement (§ 4) still works mathematically but emits a `logging.warning` flagging that the duplication factor exceeds 1. |
| `corpus.n_samples < window` | `ValueError`. |
| `corpus.frequencies_hz` length ≠ 4 | `ValueError` — the selector dimensionality is hard-coded to 4 in `constants.py`; supporting other counts is an out-of-scope architectural change. |
| `corpus.clean.shape[0] != 4` | `ValueError`, same reason. |
| Sampling seed is `None` | `TypeError`. The SDK supplies the seed if config does not. |

---

## 11. Acceptance Criteria

The dataset-construction subsystem ships when:

- AC-DS-1: All tests in § 9 pass — explicitly: `T-DS-04` through `T-DS-10`, `T-DS-11'`, `T-DS-12` through `T-DS-18`. (IDs `T-DS-01..03` were retired in v1.01 with the removal of `compute_split_ranges`.)
- AC-DS-2: `dataset.py` is ≤ 150 LOC (target ~100 in v1.01, down from ~120 in v1.00).
- AC-DS-3: Coverage on the module is ≥ 95%.
- AC-DS-4: `ruff check` clean.
- AC-DS-5: A short visualization notebook cell shows one example from each split: `w_noisy` overlaid on the underlying `noisy_sum`, with the selected `w_clean` highlighted. Figure committed to `assets/figures/dataset_examples.png`.
- AC-DS-6: Building Block docstrings on `WindowExample`, `SplitDatasets`, `WindowDataset`, `DatasetConfig`, and every public function.
- AC-DS-7: An integration test `tests/integration/test_sdk_smoke.py` exercises `SDK.generate_corpus()` → `SDK.build_dataset()` → `len(datasets.train)` end-to-end.
- AC-DS-8: `WindowDataset.meta(i)` is idempotent across calls and returns `{"t_0", "k", "split_name"}` with valid values — verified by `T-DS-18`.
- AC-DS-9: An analysis notebook cell shows overlaid `t_0` histograms for the train, val, and test splits, demonstrating they sample from the same distribution. Figure committed to `assets/figures/dataset_t0_histograms.png`. This is the visual counterpart to the framing sentence in § 5 / `ADR-016`.

---

## 12. Alternatives Considered (high-level — full text in ADRs)

- **Disjoint `t_0` ranges with `W − 1` sample buffer.** Rejected. Stationarity makes train and test distributions identical regardless of how `t_0` is partitioned, so disjoint ranges would not measure a generalization gap; they would be cosmetic. Full reasoning in `ADR-016`.
- **"Shoulder-disjoint" ranges (disjoint in `t_0` only, sharing covered samples).** Rejected as a half-measure. If disjointness were the goal, it would have to be enforced in the sample space, not just the start-index space. This was a v1.00 draft formulation that was caught in review and is documented in `ADR-016` as the cautionary alternative.
- **Different corpora per split.** Rejected. The brief specifies a single 10-vector corpus that is the substrate for sampling; generating per-split corpora would contradict the brief and change the experimental design.
- **Sampling without replacement everywhere.** Rejected. With-replacement gives true iid draws from the shared pool (the property `ADR-016` relies on); without-replacement would slightly distort the marginal at the tail and break the iid framing.
- **One-hot stored as `int64` class index.** Rejected. Float-typed one-hot composes seamlessly with the selector-broadcast scheme in RNN/LSTM input construction (`PRD_models.md`).
- **Returning model-shaped tensors directly from the Dataset.** Rejected. Couples the Dataset to model kind. The current design lets a single Dataset feed all three architectures.
- **Materializing all 30 000 windows up front in a `(30000, 10)` tensor.** Rejected. On-the-fly slicing through the index table is cheap (~µs per item), keeps memory low, and matches PyTorch's `Dataset` idiom.

---

## 13. Decisions Locked at v1.01 Review

The three open questions raised in v1.00 have been resolved:

1. **DataLoader shuffling.** Train shuffles each epoch; val and test iterate in a deterministic order. Locked in `ADR-017-dataloader-shuffling.md`. Implementation lives in the (yet-to-be-written) `PRD_training_evaluation.md`; this PRD references the ADR and contractually requires the val/test order to be reproducible across runs.
2. **Debug metadata.** `WindowDataset` exposes `meta(i: int) -> dict` returning `{"t_0", "k", "split_name"}`. The method is required to be idempotent; verified by `T-DS-18` and pinned by acceptance criterion `AC-DS-8`.
3. **Input normalization.** None in v1.00. Signals are already O(1) (amplitudes = 1, phase noise bounded). If the smoke benchmark `EXP-001` (M3+M4) surfaces optimizer instability that traces back to input scale, the decision is reopened in a new `ADR-018` at that time. Until that trigger fires, no `ADR-018` is pre-planned and no normalization layer is added.

---

## 14. References

- `HOMEWORK_BRIEF.md` § 4 (dataset construction).
- `docs/PRD.md` v1.01 § 13 item 1 (dataset size locked).
- `docs/PLAN.md` v1.02 § 9 (config schema — `split_strategy` field removed, ADR-016 / ADR-017 folded in).
- `docs/PRD_signal_generation.md` v1.01 (Corpus contract; stationarity argument).
- `docs/adr/ADR-004-dataset-size.md` (M0, written).
- `docs/adr/ADR-016-random-sampling-stationary.md` (M0, written alongside this v1.01).
- `docs/adr/ADR-017-dataloader-shuffling.md` (M0, written alongside this v1.01).

---

## 15. v1.02 Follow-up Checklist

Items identified during PRD authoring to be addressed in the PLAN.md v1.02 sweep. Status reflects completion after that sweep (2026-05-01).

- [x] **PLAN.md § 8 — Model package LOC ceilings tightened.** `__init__.py` 10 → 30; `fc.py`, `rnn.py`, `lstm.py` 80 → 50 each; total 310 → 240. Reflects observed PyTorch-delegate footprint with headroom. (`PRD_models.md § 8`, `PLAN.md v1.02`)
- [x] **PLAN.md § 8 — `dataset.py` LOC ceiling updated.** ~120 → ~100. (`PRD_dataset_construction.md § 8`, `PLAN.md v1.02`)
- [x] **PLAN.md § 9 — `split_strategy` field removed from config schema.** ADR-016 fixes a single strategy; the field was a v1.00 placeholder. (`ADR-016`, `PRD_dataset_construction.md § 3.1`)
- [x] **PLAN.md § 13 — ADR-016 and ADR-017 promoted to "M0, written".** Both written alongside `PRD_dataset_construction.md` v1.01.
- [x] **PLAN.md § 13 — ADR-018 (input normalization) listed as conditional/deferred.** Not pre-planned; triggered only if `EXP-001` surfaces optimizer instability traceable to input scale. (`PRD_dataset_construction.md § 13 item 3`)
- [x] **All eight M0 upfront ADRs written.** ADR-001, 002, 004, 005, 009, 012, 014, 015 now exist in `docs/adr/`.

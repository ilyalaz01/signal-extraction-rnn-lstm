# ADR-017: DataLoader shuffling — train stochastic, val/test deterministic

**Status:** Accepted (2026-04-30)
**Supersedes:** none.
**Owners:** training subsystem (`PRD_training_evaluation.md`, scope-touching `PRD_dataset_construction.md` § 13).

## Context

Three PyTorch `DataLoader`s feed the training loop, one per split. Their iteration order interacts with two reproducibility concerns:

1. **Train.** SGD convergence benefits from re-shuffling each epoch. Without shuffling, batches are repeatedly drawn in the same order and the optimizer can pick up trajectory artifacts that have nothing to do with the loss landscape.
2. **Val / test.** The metric of interest is MSE aggregated over the full split. Aggregate MSE is invariant to iteration order, but per-batch logging, partial-pass evaluation (e.g. evaluating the first `k` batches as a fast diagnostic), and minibatch-level diagnostics are not. Stable, deterministic order makes those reproducible across epochs and across runs.

Reproducibility is a load-bearing property of this project (`PLAN.md` § 11): one `runtime.seed` should fully determine all randomness in data and order of presentation.

## Decision

- **Train DataLoader:** `shuffle=True`. Reshuffles each epoch using PyTorch's default behavior (driven by the global RNG, which is itself seeded from `runtime.seed`).
- **Val / Test DataLoaders:** `shuffle=False`. Iterates in the order the index table was generated.

The val and test index tables are themselves deterministic — `WindowDataset` precomputes its `(t_0, k)` table from a fixed split-specific seed (see `PRD_dataset_construction.md` § 4 and § 6). Therefore, across two runs with the same `runtime.seed`, the val and test minibatch sequences are byte-identical: per-batch val loss curves can be subtracted to measure model differences, and partial-pass evaluation is reproducible.

## Alternatives Considered

**(a) `shuffle=True` for all three.**
*Rejected.* Val MSE per batch becomes a noisier signal across epochs, complicating early-stopping decisions and minibatch-level diagnostics. Aggregate MSE is unchanged but reproducibility of finer-grained signals is lost for no benefit.

**(b) `shuffle=False` for all three.**
*Rejected.* Train benefits from epoch-level reshuffling for SGD convergence; removing it for the sake of cross-split symmetry costs more than it saves.

**(c) `shuffle=True` for val/test with a fixed `torch.Generator` seed.**
*Rejected for simplicity.* This produces effectively the same outcome as `shuffle=False` (a deterministic order across epochs) at the cost of an additional `Generator` object. `shuffle=False` already gives a deterministic, reproducible order — we use that.

## Consequences

**Positive:**
- Val/test minibatch sequences are reproducible across epochs and across runs (with the same seed).
- Train benefits from per-epoch reshuffling for SGD convergence.
- Implementation uses stock PyTorch `DataLoader` flags only; no custom samplers or generators.

**Negative:**
- The val and test order is fixed at index-table-generation time. If the index table is regenerated with a different `runtime.seed`, the order changes — but that is the desired behavior (one knob controls all data-related randomness).
- An ablation that wants to randomize val order across epochs must explicitly override this default. That is a v2 concern, not a v1.00 concern.

## References

- `docs/PRD_dataset_construction.md` v1.01 § 6, § 13.
- `docs/PRD_training_evaluation.md` (to be written) — DataLoader construction.
- `docs/PLAN.md` v1.01 § 11 (reproducibility / seeding strategy).
- PyTorch docs: `torch.utils.data.DataLoader` `shuffle` argument.

# ADR-004: Dataset size — 30 000 / 3 750 / 3 750

**Status:** Accepted (2026-05-01)
**Supersedes:** none.
**Owners:** dataset-construction subsystem (`PRD_dataset_construction.md § 3.1`).

## Context

`HOMEWORK_BRIEF.md § 4.3` leaves dataset size open: "suggested starting point: 20 000 – 50 000 training windows, 80/10/10 split." The project needs a concrete size that is large enough to expose differences between FC, RNN, and LSTM on a per-frequency MSE basis and small enough to train in a reasonable time on CPU.

Key constraints:
- The per-frequency evaluation (`PRD_training_evaluation.md § 7.2`) filters the test set by channel; each channel gets ≈ 25% of test examples. A stable per-frequency MSE estimate needs several hundred examples per channel per model.
- The multi-seed grid (ADR-007, deferred to M5) will run each (model, seed) pair; total training budget is `n_seeds × 3 models × n_train_examples` forward-backward passes.
- The full `(t_0, k)` sampling pool has 39 964 unique slots (9 991 start indices × 4 channels). The train split of 30 000 examples is within the pool size, so with-replacement sampling produces genuine stochastic variety.

## Decision

**30 000 train / 3 750 val / 3 750 test** (37 500 total; exact 80/10/10 split).

Per-channel breakdown at test time: ≈ 937 examples per sinusoid (3 750 / 4), with binomial std ≈ 26. This gives a stable MSE estimate for the thesis evaluation.

## Alternatives Considered

**(a) Smaller dataset (e.g., 10 000 / 1 250 / 1 250).**
*Rejected.* At 312 test examples per channel, seed-to-seed variance in per-frequency MSE would be high enough to obscure the LSTM vs. RNN signal. The multi-seed grid (ADR-007) requires meaningful per-seed MSEs to measure standard deviation; a small dataset amplifies noise in each individual run.

**(b) Larger dataset (e.g., 80 000 / 10 000 / 10 000).**
*Rejected.* Training time on CPU grows linearly with training set size. The three-way baseline (EXP-001) must complete in a session; 80 000 examples × 3 models × 30 epochs would push that boundary without a material accuracy improvement on a regression of this scale (sinusoidal signals are low-entropy; the model saturates well before 80k examples).

**(c) Fractional split instead of absolute counts.**
*Rejected.* Under the random-sampling design (ADR-016), splits draw from the full `t_0` pool independently — there is no pool to partition by fraction. Absolute counts are the natural unit; `train_fraction` would be a derived quantity with no implementation role.

## Consequences

**Positive:**
- 30 000 train examples with Adam at batch size 256 gives ~117 steps per epoch; at 30 epochs, training fits comfortably on CPU within minutes.
- 937 test examples per frequency channel gives a stable MSE estimate: coefficient of variation ≈ 3% assuming normally distributed per-example losses.
- The 80/10/10 ratio matches the HOMEWORK_BRIEF suggestion and is a recognizable industry convention.

**Negative:**
- None material for this scale of task.

## References

- `HOMEWORK_BRIEF.md § 4.3` (dataset size left open).
- `docs/PRD_dataset_construction.md` v1.01 § 3.1 (setup parameters table).
- `docs/adr/ADR-016-random-sampling-stationary.md` (why absolute counts, not fractions).
- `docs/PLAN.md` v1.02 § 13.1 (ADR index).

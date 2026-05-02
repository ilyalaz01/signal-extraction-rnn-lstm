# ADR-007: Three seeds per (model, frequency) cell for v1.00

**Status:** Accepted (2026-05-02). Promoted from "deferred" before EXP-001 launch.
**Supersedes:** none.
**Owners:** training & evaluation subsystem (`PRD_training_evaluation.md` § 8.3 / § 8.4).

## Context

`PRD.md` v1.01 AC-2 requires **at least three seeds with confidence intervals** for the comparative study. `PRD_training_evaluation.md` § 8.3 ties the 10% practical-significance threshold to the observed seed-to-seed standard deviation: "the threshold should be re-validated as ≥ 2× the observed seed-std for the most-variable model." § 8.5 also marks the single-run qualifier as **mandatory** in any single-seed README — meaning a one-seed EXP-001 cannot satisfy the thesis evaluation protocol without explicit re-running later.

EXP-001 is the baseline three-way comparison (FC / RNN / LSTM × 4 frequencies). Without a multi-seed grid:

- AC-2 is unmet.
- The 10% threshold stays "provisional" indefinitely.
- The Spearman ρ test in § 8.6 has only four data points (one per frequency); the test is honest but its variance estimate is implicit.

Two alternatives were on the table for v1.00 — promote ADR-007 now and run a multi-seed EXP-001, or run a single-seed EXP-001 and defer to a follow-up grid.

## Decision

**Three seeds per (model, frequency) cell**, fixed before EXP-001 runs. Seed values: `[1337, 1338, 1339]` (config-default seed plus two consecutive integers; documented choice, no cherry-picking).

For EXP-001 this means **9 training runs total** (three model kinds × three seeds), each on the full default config (30 000 train / 3 750 val / 3 750 test, 30 epochs). Each run produces a `results.json` and a pickled `result.pkl` in its own ADR-014 run directory. The analysis notebook computes per-cell mean and std across the three seeds and reports them in the EXP-001 comparison table.

## Alternatives Considered

**(a) One seed.** Rejected. Fails AC-2 explicitly. Forces "single run" qualifier into the README permanently. Makes the 10% threshold un-validatable. The compute saving (3× fewer runs) is small in absolute terms (single-run wall-clock is minutes, not hours).

**(b) Five seeds.** Rejected for v1.00. Two extra runs per cell (six extra total) yield a marginally tighter std estimate — at N=3 the std estimator has ~50% relative error, at N=5 it has ~37%. The improvement does not justify 67% more compute when the bigger sources of uncertainty (single ablation per noise parameter, single corpus sample) are still N=1. If the observed seed-std at N=3 is large enough that a tighter estimate is needed, ADR-007 is reopened and a follow-up grid is run.

**(c) Three seeds for some models, fewer for others.** Rejected. Asymmetric N across model kinds breaks the per-cell apples-to-apples comparison the thesis depends on. All three models pay the same compute cost for the same statistical resolution.

## Consequences

**Positive:**
- Satisfies `PRD.md` AC-2 with the minimum sufficient seed count.
- Provides a per-cell std that lets `PRD_training_evaluation.md` § 8.3 finalize the 10% threshold (or replace it with an evidence-based number) instead of leaving it "provisional."
- Enables the README "Thesis evaluation" section to drop the "single run" qualifier and quote per-cell mean ± std.
- Spearman ρ at p < 0.10 (§ 8.6) is computed against per-cell means, with the seed-std giving an estimate of cell uncertainty that goes into the figure caption.

**Negative:**
- 3× compute relative to a single-seed baseline. With observed per-run wall-clock ~3–10 min CPU, total budget is ~30–90 min for the full 9-run grid. Within the 3-hour ceiling agreed for EXP-001.
- The std estimate at N=3 is itself noisy (~50% relative error). The threshold rule of thumb in § 8.3 ("threshold ≥ 2× std") therefore has its own uncertainty. The mitigation is to reopen this ADR and add seeds if the observed std looks load-bearing.

## References

- `docs/PRD.md` v1.01 § AC-2 (multi-seed requirement).
- `docs/PRD_training_evaluation.md` v1.01 § 8.3 (10% threshold + seed-std re-validation), § 8.5 ("single run" qualifier), § 8.6 (Spearman ρ).
- `docs/PLAN.md` v1.02 § 13 (ADR index — entry promoted from deferred).
- `docs/experiments/EXP-001-baseline-3seeds.md` (this ADR's runtime artefact).

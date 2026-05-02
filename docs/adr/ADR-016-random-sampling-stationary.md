# ADR-016: Random t_0 sampling without disjoint splits

**Status:** Accepted (2026-04-30)
**Supersedes:** none.
**Owners:** dataset-construction subsystem (`PRD_dataset_construction.md` § 5).

## Context

The corpus is a 10-second multi-sine signal sampled at 1 kHz (`N = 10 000` samples) with window length `W = 10`, giving `N - W + 1 = 9 991` valid start indices `t_0 ∈ [0, 9 990]`. Combined with the 4-channel selector, the full sampling pool has `39 964` unique `(t_0, k)` slots. Split sizes are inherited from `ADR-004`: 30 000 / 3 750 / 3 750 (train / val / test).

The signal generation process is **stationary**: every `t ∈ [0, N)` is governed by the same generative law — sum of four fixed-frequency, fixed-amplitude sinusoids with stationary i.i.d. amplitude and phase noise (`PRD_signal_generation.md` § 3). The marginal distribution over windows is therefore identical for every `t_0`.

We must decide how `t_0` is sampled across the three splits. A v1.00 draft of this decision proposed disjoint `t_0` ranges (train: `[0, 7 990]`, val: `[7 991, 8 990]`, test: `[8 991, 9 990]`) on the grounds that this would prevent train/test "leakage". Review caught two problems with that draft:

1. **The disjointness was illusory.** The proposed boundaries were disjoint in `t_0` but not in *covered samples* — a train window starting at `t_0 = 7 990` covers samples `[7 990, 7 999]`, overlapping a val window starting at `t_0 = 7 991` over nine of its ten samples.
2. **The motivation was misframed.** Stationarity means train and test draws come from *the same distribution*, so there is no generalization gap to measure across `t_0` ranges. The "leakage" framing only applies when the underlying process is non-stationary or when training memorizes specific `t_0` values — neither holds here.

## Decision

`t_0` is drawn iid uniformly from `[0, N - W]` in **all three splits**. Splits differ only in:

1. **Example count** (`n_train = 30 000`, `n_val = 3 750`, `n_test = 3 750`).
2. **RNG seed** — `sampling_seed` is spawned into three independent child seeds via `np.random.SeedSequence(sampling_seed).spawn(3)` so that train, val, and test draws are statistically independent.

> **Framing sentence (reproduced verbatim in `README.md` § Methodology when M6 ships):**
> Splits exist by example count (30 000 / 3 750 / 3 750) for reproducibility and reporting, but `t_0` is drawn iid from the full pool `[0, N - W]` in all three splits because the underlying signal process is stationary; disjoint `t_0` ranges would not measure a generalization gap.

## Alternatives Considered

**(a) Disjoint `t_0` ranges with a `W - 1` sample buffer.**
*Rejected.* Stationarity makes the train and test distributions identical regardless of how `t_0` is partitioned, so a "generalization gap" is not what MSE on test measures here; the splits are reporting partitions, not distribution-shift measurements. The hygiene argument ("at least it looks like best practice") is outweighed by the simplicity argument: a partition that is statistically meaningless is engineering theater.

**(b) "Shoulder-disjoint" ranges (disjoint in `t_0` only, sharing covered samples).**
*Rejected as a half-measure.* If disjointness were the goal, it would have to be enforced in the sample space (with a `W - 1` buffer between split ranges), not just the start-index space. This was the v1.00 draft formulation and was caught in review. We do not adopt a third option intermediate between (a) and the chosen decision.

**(c) Different corpora per split.**
*Rejected.* The brief specifies a single 10-vector corpus that is the substrate for sampling (`HOMEWORK_BRIEF.md` § 4). Generating multiple corpora would contradict the brief and change the experimental design without justification.

## Consequences

**Positive:**
- Simpler code in `services/dataset.py` — no boundary arithmetic, one `rng.choice` call per split, no `compute_split_ranges` function. The dataset module's LOC budget falls comfortably under 150.
- The off-by-one boundary bug class (the v1.00 mistake) is eliminated by construction.
- `T-DS-13` (KS test of `t_0` against `Uniform[0, N - W]`) and the new `T-DS-11'` (KS equivalence between `train.t_0` and `test.t_0`) assert what is actually true rather than approximating uniformity within a sub-range.
- MSE on `test` is an unbiased estimator of expected MSE on the data distribution.

**Negative:**
- A reviewer skimming for "is there a split" might miss the reasoning. Mitigation: the framing sentence above is repeated verbatim in `README.md` (M6) and in this ADR; the analysis notebook includes an overlaid `t_0`-histogram figure (`AC-DS-9`) as visual proof of equal-distribution sampling.
- If a future ablation introduces a non-stationary signal (e.g., time-varying frequencies, drifting noise statistics), the splits will need to be reopened. That ablation triggers a successor ADR; v1.00 is not designed for that case.

## References

- `HOMEWORK_BRIEF.md` § 4 (dataset construction contract).
- `docs/PRD_dataset_construction.md` v1.01 § 5 (split strategy).
- `docs/PRD_signal_generation.md` v1.01 § 3 (stationarity of the generative process).
- `docs/adr/ADR-004-dataset-size.md` (inherited 30k / 3.75k / 3.75k counts).

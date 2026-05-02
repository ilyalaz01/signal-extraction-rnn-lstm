# ADR-003: Selector-broadcast input scheme for RNN/LSTM

**Status:** Accepted (2026-04-30)
**Supersedes:** none.
**Owners:** models subsystem (`PRD_models.md` § 2).

## Context

A training example consists of a 4-dim one-hot selector `C` and a 10-sample noisy window `W`; the target is the corresponding 10-sample clean window. `HOMEWORK_BRIEF.md` § 5.1 fixes the input shapes:

- **FC** receives a flat 14-vector `[C ⊕ W]`.
- **RNN / LSTM** receive a sequence of length 10.

The brief leaves open *how* `C` is fed to the recurrent models. The choice is non-trivial: it determines whether RNN and LSTM see the selector at every step, only at the start, or via a structural mechanism that differs between architectures. The choice also determines whether FC, RNN, and LSTM see comparable information — the core requirement for a fair comparative study (the lecturer's grading priority 1, `PRD.md` § 4).

Three injection strategies are reasonable:

1. **Broadcast.** At each timestep `t`, feed `[W[t], C[0], C[1], C[2], C[3]]`. Per-step features = 5; sequence length = 10.
2. **Hidden-state initialization.** Initialize `h_0` from a learned function `f(C)`; per-step features = 1; sequence length = 10.
3. **Prefix step.** Prepend a "selector" timestep before the 10 window samples; per-step features = 1; sequence length = 11.

## Decision

**Strategy 1 (broadcast).** At each of the 10 timesteps, the per-step feature vector is `[W[t], C[0], C[1], C[2], C[3]]`, giving per-step input size 5 and total sequence shape `(B, 10, 5)`. This applies identically to RNN and LSTM.

`HOMEWORK_BRIEF.md` § 5.2 already adopts the broadcast scheme as the canonical injection method; this ADR records the architectural reasoning.

## Alternatives Considered

**(a) Hidden-state initialization.**
*Rejected for v1.00.* Three problems:
- *Asymmetry between RNN and LSTM.* Vanilla RNN initializes `h_0` only; LSTM needs `(h_0, c_0)`. Two distinct injection mechanisms break the apples-to-apples comparison.
- *Coupling to recurrent flavor.* The injection function `f(C)` becomes part of the model's identity, and a switch from RNN to LSTM is no longer a "drop-in replacement" — it requires also redesigning `f`.
- *Reduced information rate.* The selector is seen once at `t = 0`; for a 10-step sequence with vanishing gradients (RNN) or saturating gates (LSTM under specific conditions), this can dilute the conditioning signal.

Documented as a planned ablation (`EXP-006`, post-v1.00). It may expose architecture-specific advantages once the v1.00 baseline is established.

**(b) Prefix step.**
*Rejected.* Changes the effective sequence length to 11, distorting the "context window = 10" requirement (`HOMEWORK_BRIEF.md` § 4.2, which is locked). It also confuses the RNN-vs-LSTM thesis: a longer sequence shifts the vanishing-gradient regime in ways that have nothing to do with the architectural difference being studied.

**(c) Concatenate `C` to the output of an intermediate layer.**
*Rejected.* Violates the "all three models see the same input" principle: FC and RNN/LSTM would no longer be receiving the conditioning signal at the same point in their processing pipelines, contaminating the comparison.

## Consequences

**Positive:**
- RNN and LSTM share an identical per-step input shape (`5`) and sequence shape (`(B, 10, 5)`). The reshape utility lives in `services/models/base.py` and is applied once for both models.
- FC and RNN/LSTM see the same total information about `C` at the same total rate (4 dims × 10 timesteps = 40 element-views for RNN/LSTM; 4 dims × 1 view = 4 for FC, but FC also has its full 14-vector available simultaneously). The information rate per processing step is therefore comparable.
- The fact that RNN and LSTM share their shape contract is what makes the comparison fair (`PLAN.md` § 7.2).

**Negative:**
- Broadcasting `C` at every timestep is "wasteful" in the sense that the selector is the same at every step. A more parameter-efficient design might condition the recurrent state on `C` once. Mitigation: this is a v2 concern; for a 4-dim broadcast, the cost is negligible.
- A reviewer may argue that hidden-state initialization is "the natural way" to condition recurrent models. Mitigation: § Alternatives above and `PRD_models.md` § 2 spell out why the simpler, comparison-friendly choice was made; the alternative is on the ablation roadmap, not refused.

## References

- `HOMEWORK_BRIEF.md` § 5.1, § 5.2.
- `docs/PRD.md` v1.01 § 4 (NFR-3 — uniform model interface).
- `docs/PLAN.md` v1.01 § 7.2 (per-batch shape contract).
- `docs/PRD_models.md` v1.00 § 2 (full mechanics of the broadcast scheme).

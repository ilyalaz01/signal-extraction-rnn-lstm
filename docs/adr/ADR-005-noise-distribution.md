# ADR-005: Noise distribution — Gaussian, per-sample

**Status:** Accepted (2026-05-01)
**Supersedes:** none.
**Owners:** signal-generation subsystem (`PRD_signal_generation.md § 2.2`).

## Context

The noisy sinusoid model from `HOMEWORK_BRIEF.md § 3.2` is:

$$\tilde{s}_i[t] = \bigl(A_i + \alpha \cdot \epsilon^{A}_{i}[t]\bigr) \cdot \sin\!\left(2\pi f_i t + \phi_i + \beta \cdot \epsilon^{\phi}_{i}[t]\right)$$

Two design choices are left open by the brief:

1. **Distribution** of ε_A and ε_φ: the brief says "Uniform or Gaussian; lecturer said both are acceptable."
2. **Granularity**: are ε sampled once per signal (per-signal) or independently at each timestep (per-sample)?

## Decision

**Gaussian, per-sample.** Specifically:
- $\epsilon^{A}_{i}[t] \sim \mathcal{N}(0, 1)$, iid across channels $i$ and timesteps $t$.
- $\epsilon^{\phi}_{i}[t] \sim \mathcal{N}(0, 1)$, iid across channels $i$ and timesteps $t$.
- $\alpha = 0.05$ (5% amplitude noise relative to unit amplitude).
- $\beta = 2\pi$ (full-range phase noise — lecturer's preferred setting per HOMEWORK_BRIEF § 3.2).

## Alternatives Considered

**(a) Uniform distribution (variance-matched).**
*Acceptable but not chosen.* The brief explicitly accepts Uniform. Gaussian is preferred because:
- It is the standard model for additive white Gaussian noise (AWGN) in signal processing — physically motivated for thermal and quantization noise.
- The Central Limit Theorem means that sums of many iid noise sources converge to Gaussian regardless of the individual distribution; Gaussian is therefore the more principled default for amplitude and phase perturbations.
- Future ablation `EXP-004` will compare Uniform (variance-matched: `U(-√3, √3)`) against Gaussian; the decision here fixes v1.00 and does not foreclose the ablation.

**(b) Per-signal granularity (draw ε once per channel, apply to all timesteps).**
*Rejected.* Per-signal amplitude noise is mathematically equivalent to randomly shifting the amplitude constant for the entire 10 000-sample realization. For a 10-sample window, the shift is indistinguishable from a slightly different $A_i$ — the network sees a constant-amplitude sinusoid, not one with temporal amplitude variation. This makes amplitude noise vacuous as a training-time challenge: the source of difficulty disappears in any single window. Per-sample noise, by contrast, produces moment-to-moment amplitude and phase variation that the network must implicitly smooth over, matching the spirit of the AWGN denoising task.

**(c) Per-signal phase noise (draw ε_φ once per channel).**
*Rejected.* Same reasoning as (b). A per-signal phase offset is a fixed rotation of the sinusoid for the entire corpus realization — effectively just a change of $\phi_i$. It produces no within-window variation; the network cannot use temporal structure to distinguish "noisy" from "phase-shifted." Per-sample phase noise makes every window genuinely uncertain.

## Consequences

**Positive:**
- Per-sample Gaussian noise produces a stationary random process that justifies the random-sampling split design (ADR-016): the distribution is the same at every $t_0$.
- The noise model is fully specified by three scalars ($\alpha$, $\beta$, distribution) in `config/setup.json`. Ablations change one scalar at a time (EXP-002, EXP-003, EXP-004).

**Negative:**
- With $\beta = 2\pi$, per-sample phase noise has std $2\pi$ rad, meaning instantaneous phase can swing by a full cycle each sample. This is physically extreme (more like random FM than additive noise). The lecturer explicitly endorsed "0 to 2π is more interesting," accepting this choice. The consequence for models is that high-frequency sinusoids (200 Hz, 2 cycles per window) are genuinely hard to track because the instantaneous phase is nearly unpredictable.

## References

- `HOMEWORK_BRIEF.md § 3.2` (noise model specification).
- `docs/PRD_signal_generation.md` v1.01 § 2.2 (full noise model with equations).
- `docs/PLAN.md` v1.02 § 9 (config schema: `signal.noise`), § 13.1 (ADR index).

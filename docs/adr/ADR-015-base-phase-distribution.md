# ADR-015: Base-phase distribution — [0, π/2, π, 3π/2]

**Status:** Accepted (2026-05-01)
**Supersedes:** none.
**Owners:** signal-generation subsystem (`PRD_signal_generation.md`); config schema (`PLAN.md § 9`).

## Context

Each of the four sinusoids $s_i(t) = A_i \cdot \sin(2\pi f_i t + \phi_i)$ has a base phase $\phi_i$. The homework brief does not specify the phases; choosing them is part of the assignment.

The base phases affect:
1. **Composite signal behavior at t = 0.** If all four sinusoids share the same phase, they are all near their zero-crossing simultaneously at t = 0, creating constructive interference that produces a large-amplitude artifact in windows sampled near the start of the corpus.
2. **Training difficulty.** A poorly chosen phase set can make one sinusoid trivially identifiable (e.g., one is always at its peak while others are at zero) or introduce symmetry that reduces the task difficulty.
3. **Corpus seed coupling.** If phases are drawn randomly, the corpus seed determines the phase semantics, making it harder to reason about the signal independently of randomness.

## Decision

**`[0, π/2, π, 3π/2]`** — four phases evenly distributed across the unit circle, stored in `config/setup.json` as `["0", "pi/2", "pi", "3*pi/2"]` (string expressions parsed at load time by `shared/config.py`).

This gives `s_1` a sine-like start (rising from zero), `s_2` a cosine-like start (at its peak), `s_3` a negative-sine start (falling from zero), and `s_4` a negative-cosine start (at its trough). The four sinusoids are in quadrature at t = 0.

## Alternatives Considered

**(a) All-zero phases `[0, 0, 0, 0]`.**
*Rejected.* At t = 0, all four sinusoids are at a zero-crossing simultaneously. Their sum $S(0) = 0$ and their time derivatives are all positive — a degenerate start that may produce large-amplitude behavior immediately after t = 0 (within the first window) when all sines are rising together. Windows sampled near t = 0 would have a characteristic transient not present elsewhere. The signal is *not* perfectly stationary near t = 0 in a statistical sense if phases align — the stationarity argument underlying `ADR-016` is stronger when phase staggering prevents start-time artifacts.

**(b) Random phases per corpus seed.**
*Rejected.* Phase values would differ across seeds, coupling the "what does the signal look like" question to the "which seed are we running" question. Debugging and describing the corpus would require always citing the seed. Fixed phases keep the signal description fully self-contained in `config/setup.json`.

**(c) Phases derived from frequency ratios (e.g., `[0, 2π·f_1/f_4, 2π·f_2/f_4, 2π·f_3/f_4]`).**
*Rejected.* Arbitrary and hard to interpret. The uniform-circle choice has a clear geometric rationale and is easy to verify by inspection.

## Consequences

**Positive:**
- At t = 0, the four sinusoids span all four quadrants — no constructive interference artifact.
- The phases are simple fractions of π, easy to verify in the test suite (T-SG-* in `PRD_signal_generation.md § 8`).
- Stored as angle-expression strings (`"pi/2"` not `1.5707963...`) for human readability and round-trip exactness.

**Negative:**
- The quadrature phasing means that, at t = 0 exactly, $s_2$ is at its maximum while $s_4$ is at its minimum. This is a specific feature of the corpus, not a bug, but it should be noted in the analysis notebook caption for the signal overview figure (`AC-SG-5`).

## References

- `docs/PRD_signal_generation.md` v1.01 § 3 (signal parameters table), § 11 (alternatives).
- `docs/PLAN.md` v1.02 § 9 (config schema: `signal.phases_rad`), § 13.1 (ADR index).
- `docs/adr/ADR-016-random-sampling-stationary.md` (stationarity argument that quadrature phasing supports).

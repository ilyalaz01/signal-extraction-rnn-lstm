# PRD — Signal Generation

> **Document version:** 1.01
> **Status:** Approved 2026-04-30 (with corrections to § 2.2 noise model and minor polish).
> **Owns:** the ten-vector signal corpus that is the substrate for everything downstream.
> **Companion:** `HOMEWORK_BRIEF.md` § 3, `docs/PRD.md` v1.01, `docs/PLAN.md` v1.01, `docs/adr/ADR-005-noise-distribution.md`, `docs/adr/ADR-015-base-phase-distribution.md`.

This is the dedicated PRD for the **signal generation** subsystem. It defines the mathematical specification, contract, and test plan for one Python module (`services/signal_gen.py`) and the dataclasses it produces. Anything that touches windows, splits, batching, or model inputs is **out of scope** here — that lives in `PRD_dataset_construction.md`.

---

## 1. Purpose

Produce, in a single deterministic call, the ten signal vectors required by the homework brief:

```
s_1, s_2, s_3, s_4              (clean base sinusoids, length 10 000)
s̃_1, s̃_2, s̃_3, s̃_4              (noisy base sinusoids, same length)
S      = s_1 + s_2 + s_3 + s_4   (clean sum)
S̃      = s̃_1 + s̃_2 + s̃_3 + s̃_4   (noisy sum — note: sum of noisy components)
```

Downstream code (`services.dataset`) cuts windows from these vectors. Nothing in `services.signal_gen` knows about windows, batches, or selectors.

---

## 2. Theoretical Background

### 2.1 The base model

Each base sinusoid is

$$
s_i(t) = A_i \cdot \sin(2\pi f_i t + \phi_i), \qquad t = 0, \tfrac{1}{f_s}, \tfrac{2}{f_s}, \dots, \tfrac{N-1}{f_s}
$$

with sampling rate $f_s = 1000$ Hz, duration 10 s, hence $N = 10\,000$ samples per vector.

### 2.2 The noise model

Per HOMEWORK_BRIEF § 3.2, noise is injected **inside** the sinusoid, not added to its output. **Both** noise terms are drawn **per sample, independently for each channel and each timestep**:

$$
\tilde{s}_i[t] = \bigl(A_i + \alpha \cdot \epsilon^{A}_{i}[t]\bigr) \cdot \sin\!\left( 2\pi f_i t + \phi_i + \beta \cdot \epsilon^{\phi}_{i}[t] \right)
$$

- $\epsilon^{A}_{i}[t] \sim \mathcal{N}(0, 1)$, iid across $i$ and $t$.
- $\epsilon^{\phi}_{i}[t] \sim \mathcal{N}(0, 1)$, iid across $i$ and $t$.
- $\alpha = 0.05$ — amplitude noise strength. With $\epsilon^{A}_{i}[t] \sim \mathcal{N}(0,1)$ and unit amplitude, the additive amplitude noise has std $\alpha = 0.05$ — i.e. **5% of unit amplitude**. Justified in § 7.1.
- $\beta = 2\pi$ — phase noise strength. With $\epsilon^{\phi}_{i}[t] \sim \mathcal{N}(0,1)$, the additive phase term has std $\beta = 2\pi$ rad. The lecturer's "0 to 2π is more interesting" choice, justified in § 7.2.
- Distribution: **Gaussian**. The brief allows Uniform; Gaussian was chosen in ADR-005.

The brief's formula is symmetric in $\epsilon^A$ and $\epsilon^\phi$, and the lecturer's verbal notes specified noise *strength* (percentages), not *granularity*. Per-signal (rather than per-sample) amplitude noise would be mathematically equivalent to redrawing the constant amplitude $A_i' = A_i + \alpha \delta$ — there would be no temporal noise to denoise, just a constant offset. The model above produces genuine sample-level corruption that the network must learn to attenuate.

### 2.3 The composite signals

$$
S(t) = \sum_{i=1}^{4} s_i(t), \qquad \tilde{S}(t) = \sum_{i=1}^{4} \tilde{s}_i(t)
$$

The fact that $\tilde{S}$ is the **sum of noisy components** (not noise added to $S$) is a non-trivial constraint from the brief. Adding noise to the sum would erase the per-component structure; here, each noisy component carries its own independent noise.

---

## 3. Setup, Inputs, Outputs (Building Block format — guidelines § 15.1)

### 3.1 Setup parameters (from `config/setup.json` § `signal`)

| Field | Type | Default | Constraint |
| --- | --- | --- | --- |
| `fs` | int | `1000` | > 0; HOMEWORK_BRIEF locks 1000. |
| `duration_s` | int | `10` | > 0; HOMEWORK_BRIEF locks 10. |
| `frequencies_hz` | list[float] | `[2, 10, 50, 200]` | exactly 4 strictly positive distinct values. See § 4. |
| `amplitudes` | list[float] | `[1.0, 1.0, 1.0, 1.0]` | exactly 4 positive values. |
| `phases_rad` | list[str expr] | `["0", "pi/2", "pi", "3*pi/2"]` | exactly 4. Parsed by `shared/config.py`. See ADR-015. |
| `noise.alpha` | float | `0.05` | ≥ 0. Setting to 0 disables amplitude jitter. |
| `noise.beta` | str expr | `"2*pi"` | ≥ 0. Setting to 0 disables phase jitter. Parsed by `shared/config.py`. |
| `noise.distribution` | str | `"gaussian"` | one of `{"gaussian", "uniform"}`. |

Plus a single `seed` (int) drawn from `runtime.seed`. The same seed → exactly the same corpus, on the same device.

### 3.2 Inputs

The module takes a parsed config object (`SignalConfig` dataclass) plus a seed. There is no other runtime input — no files, no network. This is what makes the module trivially testable.

### 3.3 Outputs

A single `Corpus` dataclass:

```python
@dataclass(frozen=True)
class Corpus:
    """
    Input:  produced by services.signal_gen.generate_corpus(config, seed)
    Output: this object — consumed by services.dataset
    Setup:  carries the SignalConfig and seed it was generated from, for provenance
    """
    fs: int                              # sampling rate (Hz)
    n_samples: int                       # = fs * duration_s, e.g. 10 000
    frequencies_hz: tuple[float, ...]    # length 4
    clean: np.ndarray                    # shape (4, N), float32
    noisy: np.ndarray                    # shape (4, N), float32
    clean_sum: np.ndarray                # shape (N,), float32
    noisy_sum: np.ndarray                # shape (N,), float32
    config: SignalConfig                 # the exact config used (provenance)
    seed: int                            # the exact seed used (provenance)
```

Shapes and dtype are part of the contract and are checked by tests. `float32` rather than `float64`: the downstream models train in `float32`; using the same dtype upstream avoids silent casts.

---

## 4. Frequency Set — Justification (folds ADR-008)

The frequency set `[2, 10, 50, 200] Hz` is locked in this PRD (`docs/PRD.md` § 13 item 4). It is recorded here, not as a separate ADR, because it has no architectural consequence: the SDK contract, component boundaries, and tensor shapes do not depend on the specific frequencies chosen — only on the count being 4. Changing one frequency to another is a config-only change; changing the count is the architectural change, and that triggers an ADR.

The set is justified relative to the 10-sample window at $f_s = 1000$ Hz (window = 10 ms):

| $f_i$ | Period (samples) | Periods per window | Regime |
| --- | --- | --- | --- |
| 2 Hz   | 500 | 0.02 | Very low — window covers 1/50 of a cycle. Almost flat within the window. |
| 10 Hz  | 100 | 0.1  | Low — window covers 1/10 of a cycle. Locally near-linear. |
| 50 Hz  | 20  | 0.5  | Mid — half a cycle visible. |
| 200 Hz | 5   | 2.0  | High — two full cycles visible. |

The thesis under test (HOMEWORK_BRIEF § 2) is: RNN should win on 200 Hz (local structure within window); LSTM should win on 2 Hz (the window is so flat that the network cannot disambiguate amplitude from low-frequency phase without retaining longer-term context). The set is intentionally constructed to span both regimes.

If results turn out flat (i.e. all three architectures perform similarly across the set), this PRD will gain a § labeled "Adjustment after EXP-001" listing alternative frequency sets to try (e.g. `[1, 5, 100, 400]`).

---

## 5. Numerical Considerations

### 5.1 Time grid

Use a single shared time grid `t = arange(N) / fs` (float64 internally for precision, cast to float32 for storage). Per-sinusoid recomputation of `t` is wasteful; one allocation, four broadcasts.

### 5.2 Phase wrapping

Per-sample $\phi_i + \beta \epsilon_\phi^{(i)}$ values are passed unwrapped into `np.sin`, which is $2\pi$-periodic and handles wrapping internally. Wrapping the values into `[0, 2π)` before `sin` would be unnecessary work and would not change the result. The `beta = 2π` ceiling means the wobble in the argument can be arbitrarily large, but `sin` does not care.

### 5.3 Determinism

A single `numpy.random.Generator` is constructed from the seed inside `generate_corpus`. It draws, in a fixed order:

1. **Amplitude noise tensor** — one tensor of shape `(4, N)` filled by a single `rng.standard_normal((4, N))` call (or `rng.uniform(-sqrt(3), sqrt(3), (4, N))` when `distribution="uniform"`).
2. **Phase noise tensor** — one tensor of shape `(4, N)` from a second call of the same kind.

This order is part of the contract: changing it changes the corpus. It is documented in the `make_noisy` docstring and verified by a dedicated reproducibility test (T-SG-02).

### 5.4 Memory

10 vectors × 10 000 float32 samples ≈ 400 KB total. The intermediate noise tensors (`(4, N)` amplitude + `(4, N)` phase, both float32) add another ~320 KB during generation, then go out of scope. Negligible. No memory-mapping or lazy generation is necessary.

### 5.5 Edge effects

The 10-sample window may straddle the start (`t=0`) or end (`t=10s`) of the signal. Distributing base phases across `[0, π/2, π, 3π/2]` (ADR-015) eliminates a synchronized zero-crossing artifact at `t=0` that would otherwise contaminate windows sampled near the start.

---

## 6. Public API

`services/signal_gen.py` exposes exactly these symbols:

```python
@dataclass(frozen=True)
class SignalConfig:  # parsed/validated form of config['signal']
    fs: int
    duration_s: int
    frequencies_hz: tuple[float, ...]
    amplitudes: tuple[float, ...]
    phases_rad: tuple[float, ...]            # already evaluated (no strings)
    noise_alpha: float
    noise_beta: float                         # already evaluated
    noise_distribution: Literal["gaussian", "uniform"]

def make_clean(config: SignalConfig) -> np.ndarray:
    """
    Input:  SignalConfig.
    Output: ndarray of shape (4, N), float32 — the four clean base sinusoids.
    Setup:  pure function. No randomness. No I/O.
    """

def make_noisy(config: SignalConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Input:  SignalConfig and an RNG.
    Output: ndarray of shape (4, N), float32 — the four noisy base sinusoids.
    Setup:
        Draws two iid noise tensors of shape (4, N) in a fixed order:
            1. amplitude noise eps_A ~ <distribution>(0, 1)
            2. phase     noise eps_phi ~ <distribution>(0, 1)
        Each sample's noisy value is:
            s_tilde[i, t] = (A_i + alpha * eps_A[i, t])
                            * sin(2*pi*f_i*t + phi_i + beta * eps_phi[i, t])
        Same RNG state ⇒ bitwise-identical output.
    """

def generate_corpus(config: SignalConfig, seed: int) -> Corpus:
    """
    Input:  SignalConfig, seed.
    Output: a fully populated Corpus (clean, noisy, clean_sum, noisy_sum).
    Setup:  builds an np.random.default_rng(seed); calls make_clean and make_noisy;
            sums to produce the composite signals; bundles provenance.
    """
```

Anything else in the module is private (leading underscore). The total budget for `signal_gen.py` is ~120 LOC per `PLAN.md` § 8.

---

## 7. Sensitivity Parameters & Defaults

### 7.1 Amplitude noise $\alpha = 0.05$

A 5% perturbation of unit amplitude. Large enough that the noisy components are visibly different from the clean ones; small enough that the clean sinusoid is still the dominant signal. To be swept in a sensitivity experiment (`EXP-002` planned: $\alpha \in \{0, 0.02, 0.05, 0.10, 0.20\}$).

### 7.2 Phase noise $\beta = 2\pi$

Maximum wobble. The lecturer's recommended default ("0 to 2π is more interesting" — HOMEWORK_BRIEF § 3.2). At $\beta = 2\pi$ with Gaussian samples, the phase argument is effectively scrambled at extreme draws. The net effect on $\sin(\cdot)$ is heavy-tailed but bounded in $[-1, 1]$; the signal remains a sinusoid in shape but with a stochastic wobble in instantaneous frequency. Smaller $\beta$ values (e.g. $\pi/4$) are more "physically realistic" but make the task too easy. To be swept in `EXP-003` planned: $\beta \in \{0, \pi/8, \pi/4, \pi, 2\pi\}$.

### 7.3 Distribution choice

Gaussian per ADR-005. The lecturer accepts Uniform. A future ablation (`EXP-004` planned, low-priority) compares the two; not v1.00 critical.

---

## 8. Test Specification

All tests live in `tests/unit/test_signal_gen.py` and are written **before** implementation per the TDD requirement (`SOFTWARE_PROJECT_GUIDELINES.md` § 5.1).

### 8.1 Unit tests

| ID | Behavior | Assertion |
| --- | --- | --- |
| T-SG-01 | Shapes and dtype | `corpus.clean.shape == (4, 10000)`; same for `noisy`. `clean_sum.shape == (10000,)`. All `float32`. |
| T-SG-02 | Deterministic | Same `(config, seed)` → bitwise-equal corpora across two calls. |
| T-SG-03 | Seed sensitivity | Two different seeds produce different `noisy` arrays (with overwhelming probability — assert mean abs diff > 0). |
| T-SG-04 | `noise.alpha = 0` ⇒ noisy amplitudes equal clean amplitudes | `np.allclose(corpus.noisy[i].max(), corpus.clean[i].max(), atol=…)` per channel. |
| T-SG-05 | `noise.beta = 0` ⇒ phase trace identical | with `alpha=0` AND `beta=0`, `noisy == clean` (up to fp). |
| T-SG-06 | Sum identity (clean) | `corpus.clean_sum == corpus.clean.sum(axis=0)` exactly. |
| T-SG-07 | Sum identity (noisy) | `corpus.noisy_sum == corpus.noisy.sum(axis=0)` exactly. **This is the brief's key invariant.** |
| T-SG-08 | Noisy ≠ clean | with default config, `np.allclose(corpus.noisy, corpus.clean) == False`. |
| T-SG-09 | Frequency content (FFT smoke) | dominant FFT peak of `corpus.clean[i]` is within 1 bin of `frequencies_hz[i]`. |
| T-SG-10 | Phase distribution | `corpus.config.phases_rad` round-trips through `np.sin` correctly (no string leaked). |
| T-SG-11 | Config validation | invalid configs raise: 3 frequencies → `ValueError`; negative `alpha` → `ValueError`; unknown distribution → `ValueError`. |
| T-SG-12 | Provenance | `corpus.seed`, `corpus.config` echo the inputs. |
| T-SG-13 | Per-sample independence | With $\alpha > 0$, $\beta = 0$, the empirical std of `noisy[i] - clean[i]` over the time axis is consistent with iid per-sample noise of std $\alpha \cdot A_i$ (within $3\sigma$ Monte-Carlo tolerance), and is **not** constant across time as a per-signal model would imply. The mirror test holds for $\alpha = 0$, $\beta > 0$ on the phase residual `arcsin(noisy / amp_envelope) - arcsin(clean / amp_envelope)` evaluated on a high-amplitude region. |

### 8.2 Property tests (light)

- For any seed in a small set and any frequency in a small grid, the FFT peak of `clean[i]` matches `frequencies_hz[i]`.
- For any $\alpha \in \{0, 0.05, 0.1\}$, mean amplitude of `noisy[i]` is within $3\sigma$ of $A_i$.

### 8.3 Coverage target

`signal_gen.py` is expected to reach ≥ 95% line coverage; the module is small and pure, so the bar is higher than the project floor of 85%.

---

## 9. Edge Cases

| Case | Behavior |
| --- | --- |
| `frequencies_hz` not strictly positive | `ValueError` at config-validation time. |
| `frequencies_hz` length ≠ 4 | `ValueError`. The architectural assumption "exactly 4" lives in `constants.py`. |
| Aliasing: a frequency above $f_s / 2 = 500$ Hz | `ValueError` (Nyquist violation). Currently impossible with the locked set, but the check exists for future flexibility. |
| `seed` is `None` | `TypeError` — seeding is mandatory. The SDK supplies a seed if config does not. |
| `noise.distribution = "uniform"` | Uses `rng.uniform(-sqrt(3), sqrt(3), …)` so unit variance matches Gaussian. Documented in code. |

---

## 10. Acceptance Criteria

The signal-generation subsystem ships when:

- AC-SG-1: All tests T-SG-01 through T-SG-12 pass.
- AC-SG-2: `signal_gen.py` is ≤ 150 LOC (target ~120).
- AC-SG-3: Coverage on the module is ≥ 95%.
- AC-SG-4: `ruff check` clean on the module and its test file.
- AC-SG-5: A short visualization notebook cell renders `clean` and `noisy` for all four channels and the two sums; the figure is committed to `assets/figures/signals_overview.png` for `README.md`.
- AC-SG-6: The Building Block docstring format is used on `Corpus`, `SignalConfig`, and every public function.

---

## 11. Alternatives Considered (high-level — full text in ADRs)

- **Adding noise to the sum** rather than summing noisy components — rejected; HOMEWORK_BRIEF § 3.4 forbids it.
- **Per-signal (rather than per-sample) amplitude noise** — rejected. Per-signal amplitude jitter is mathematically equivalent to redrawing the constant amplitude and adds no temporal corruption for the network to denoise. ADR-005 records this.
- **Uniform noise distribution** — deferred to a future ablation (`EXP-004`); not the v1.00 default per ADR-005.
- **Synchronized zero phases** — rejected; ADR-015 documents the interference-artifact reasoning.
- **Lazy/streaming corpus generation** — rejected; the corpus is 400 KB. Eager allocation is simpler and fully tested.

---

## 12. References

- `HOMEWORK_BRIEF.md` § 3 (signal model).
- `docs/PRD.md` v1.01 § 13 (locked defaults).
- `docs/PLAN.md` v1.01 § 9 (config schema), § 13 (ADR index).
- `docs/adr/ADR-005-noise-distribution.md` (M0, written).
- `docs/adr/ADR-015-base-phase-distribution.md` (M0, written).
- `docs/experiments/EXP-002-alpha-sweep.md`, `EXP-003-beta-sweep.md`, `EXP-004-distribution-ablation.md` (planned, M5).

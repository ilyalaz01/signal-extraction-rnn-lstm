# Homework Brief — Signal Source Extraction with FC / RNN / LSTM

> **Course:** Software / Deep Learning project under Dr. Yoram Segal.
> **Status:** Lecturer did not distribute a written assignment. This brief is the canonical project specification, reconstructed from the live lecture and locked in by the student.
> **Companion document:** `SOFTWARE_PROJECT_GUIDELINES.md` (binding engineering ruleset by the same lecturer). Both documents are normative.

---

## 1. Project Goal in One Sentence

Train and compare three neural architectures (Fully Connected, RNN, LSTM) on a **conditional source separation** task: given a small window of a noisy summed sinusoidal signal and a one-hot selector, reconstruct the corresponding clean sinusoid window.

---

## 2. Why This Task Exists (Lecturer's Thesis)

The lecturer's central pedagogical claim, which the project must investigate empirically:

- A pure sinusoid is fully determined by ≥ 2 samples (Nyquist intuition); any signal can be decomposed into sums of sines and cosines.
- **RNN** struggles with long context windows (vanishing gradient) → it is good for **short-term memory**, i.e. **high-frequency** components where local structure is informative.
- **LSTM** maintains long-term dependencies → it should be **better for low-frequency** / longer-period components.
- **FC** is the non-temporal baseline.

The deliverable is not just code that runs — it is a **comparative study** that confirms, refines, or refutes this thesis with evidence.

---

## 3. Signal Model

### 3.1 Base sinusoid

$$
s_i(t) = A_i \cdot \sin(2\pi f_i t + \phi_i)
$$

### 3.2 Noisy sinusoid

$$
\tilde{s}_i(t) = (A_i + \alpha \cdot \epsilon_A) \cdot \sin\!\left(2\pi f_i t + \phi_i + \beta \cdot \epsilon_\phi\right)
$$

- $\epsilon_A$, $\epsilon_\phi$ — random noise (Uniform or Gaussian; lecturer said both are acceptable).
- $\alpha$ — amplitude noise strength (≈ a few percent of $A$).
- $\beta$ — phase noise strength. The lecturer explicitly said **"0 to 2π is more interesting"**; a smaller bound is allowed but must be justified.

### 3.3 Sampling parameters (fixed)

| Parameter | Value |
| --- | --- |
| Sampling rate | **1000 Hz** |
| Signal duration | **10 seconds** |
| Samples per signal | **10 000** |
| Number of base sinusoids | **4** ($s_1, s_2, s_3, s_4$) |

### 3.4 Composite signals

- **Clean sum:** $S = s_1 + s_2 + s_3 + s_4$
- **Noisy sum (the one used for input):** $\tilde{S} = \tilde{s}_1 + \tilde{s}_2 + \tilde{s}_3 + \tilde{s}_4$ — i.e. **sum of noisy components**, not noise added to the sum.

This produces **10 vectors total** in the corpus: 4 clean + 4 noisy + clean sum + noisy sum.

---

## 4. Dataset Construction

Each training example has the form:

```
input  : [ C  |  W_noisy ]
target : [ W_clean_selected ]
```

Where:

- **C** — one-hot vector of length 4 selecting which sinusoid to extract. Example: `[0, 0, 1, 0]` means "extract $s_3$".
- **W_noisy** — a window of **10 consecutive samples** from $\tilde{S}$ at a randomly chosen start index.
- **W_clean_selected** — the window of **the same 10 indices** from the clean signal $s_k$ where $k$ is the index of the `1` in $C$.

### 4.1 Sampling procedure (per example)

1. Sample $C$ uniformly at random from the 4 one-hot vectors.
2. Sample a start index $t_0 \in [0,\, 10000 - 10]$ uniformly at random.
3. Cut $W_{\text{noisy}} = \tilde{S}[t_0 : t_0 + 10]$.
4. Cut $W_{\text{clean}} = s_k[t_0 : t_0 + 10]$ where $k = \arg\max(C)$.
5. Emit the pair (input, target).

### 4.2 Context window

**Fixed at 10 samples.** This is a hard constraint from the lecturer.

### 4.3 Dataset size

Lecturer left this open. Pick a size large enough to expose differences between FC / RNN / LSTM (suggested starting point: 20 000 – 50 000 training windows, 80/10/10 split). Justify the chosen size in `README.md`.

---

## 5. Model Contract

### 5.1 Input / output shapes

- **Input dimensionality (FC):** 14 (= 4 selector + 10 window samples), flat vector.
- **Input for RNN / LSTM:** sequence of length 10. **At each timestep $t$, feed `[x_t, C]` (i.e. one window sample concatenated with the broadcast selector vector), giving input feature size 5 per step.**
- **Output:** vector of length 10 = predicted clean window.

### 5.2 Why the broadcast-C scheme

The lecturer did not specify how to inject $C$ into recurrent models. This brief fixes the broadcast scheme because:
- It keeps FC, RNN, and LSTM **comparable** — all three see the same information at the same total rate.
- It does not abuse the hidden state initialization (which would couple the experiment to a specific RNN flavor).
- It is the simplest defensible default; alternatives (initial hidden state, prefix step) are documented as future ablations.

This decision MUST be recorded as an ADR in `docs/PRD_models.md`.

---

## 6. Loss Function

Mean squared error between the predicted clean window and the ground-truth clean window:

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left\| \hat{y}_i - y_i \right\|_2^2
$$

where $\hat{y}_i$ is the model's 10-sample prediction and $y_i$ is the target clean window.

---

## 7. Models to Implement

All three are mandatory:

1. **FC (Fully Connected)** — non-temporal baseline. Input is the flat 14-vector.
2. **RNN** — vanilla recurrent network over the 10-step sequence.
3. **LSTM** — long short-term memory over the same sequence.

Architectural details (number of layers, hidden size, activations, dropout, optimizer, learning rate, scheduler, batch size, epochs) are **deliberately unspecified** by the lecturer. Picking and justifying them is part of the assignment.

---

## 8. Frequency Selection (Critical Design Choice)

The lecturer did not specify the four frequencies. Because the central thesis is "RNN ↔ high-freq, LSTM ↔ low-freq", the chosen $f_i$ MUST span both regimes relative to the 10-sample window at 1 kHz (i.e. window = 10 ms).

**Recommended starting set** (justify and possibly tune):

| Sinusoid | Frequency | Period (samples) | Regime |
| --- | --- | --- | --- |
| $s_1$ | 2 Hz | 500 | very low (≪ window) |
| $s_2$ | 10 Hz | 100 | low |
| $s_3$ | 50 Hz | 20 | medium |
| $s_4$ | 200 Hz | 5 | high (≪ window) |

Amplitudes $A_i$ and base phases $\phi_i$ are also free parameters; pick distinct values so the four components are distinguishable.

---

## 9. Open Hyperparameters

These are the student's responsibility. Each MUST be justified in `README.md` and ideally swept in a sensitivity analysis:

- Architecture depth and width per model.
- Activation functions.
- Optimizer (Adam is a safe default), learning rate, scheduler.
- Batch size, epoch count, early-stopping criterion.
- Noise distribution (Uniform vs Gaussian) and exact $\alpha$, $\beta$ values.
- Train / validation / test split sizes.
- Whether to normalize inputs and how.

---

## 10. What the Lecturer Grades On

Per the lecture, in priority order:

1. **Comparative analysis.** Convince the lecturer that you understand *when* each architecture wins, *why*, and *what breaks*. Produce graphs, tables, and ablations.
2. **README depth.** Screenshots of signals, loss curves, error plots, architecture diagrams, decisions and their rationale, failed experiments included.
3. **Architectural understanding.** Be able to explain why LSTM gates matter, why RNN gradients vanish on long context, why FC is or is not enough.
4. **Engineering hygiene.** Compliance with `SOFTWARE_PROJECT_GUIDELINES.md` (uv, Ruff, ≤ 150 lines per file, SDK-first, dedicated PRDs, ≥ 85% test coverage).

Half-finished code with a strong analysis is worth more than polished code with no insight.

---

## 11. Submission Surface

At minimum:
- `docs/PRD.md`, `docs/PLAN.md`, `docs/TODO.md`.
- Dedicated PRDs: `PRD_signal_generation.md`, `PRD_dataset_construction.md`, `PRD_models.md`, `PRD_training_evaluation.md`.
- `src/` with package layout per guidelines (SDK layer, services, shared, constants).
- `tests/unit/`, `tests/integration/` with ≥ 85% coverage.
- `notebooks/` with results analysis: signal visualizations, loss curves, per-frequency MSE breakdown, FC-vs-RNN-vs-LSTM comparison plots.
- `README.md` as full user manual + research narrative.
- `pyproject.toml`, `uv.lock`, `.env-example`, `.gitignore`.

---

## 12. Note on `ApiGatekeeper`

The guidelines mandate a centralized `ApiGatekeeper` for all external API calls. **This project performs no external API calls** (training is fully local with PyTorch). The requirement is therefore vacuously satisfied. Document this explicitly in `docs/PLAN.md` with a short ADR rather than implementing a stub gatekeeper that wraps nothing.

---

## 13. Workflow Reminder

Per the guidelines, **no code is written before all PRDs and the PLAN are drafted and approved.** The first phase of work is purely documentation. Only after that does TDD on `src/` begin.

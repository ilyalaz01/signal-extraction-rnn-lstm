# PRD — Signal Source Extraction with FC / RNN / LSTM

> **Document version:** 1.01
> **Status:** Approved 2026-04-30 (with corrections to § 10, NFR-2, NFR-9, § 13).
> **Author:** ilyalaz01@gmail.com
> **Course:** Software / Deep Learning project under Dr. Yoram Segal.
> **Companion documents:** `HOMEWORK_BRIEF.md` (canonical task spec), `SOFTWARE_PROJECT_GUIDELINES.md` (engineering ruleset), dedicated PRDs in `docs/PRD_*.md`, ADRs in `docs/adr/`.

This is the top-level Product Requirements Document. It defines **what** the project must deliver and **why**. It does not contain mechanism-level specifications (see dedicated PRDs) and it does not contain architectural decisions (see `docs/adr/`). When a topic could plausibly live here or in a dedicated artifact, prefer the dedicated artifact and link to it.

---

## 1. Project Overview

### 1.1 One-line summary
A reproducible, deeply documented comparative study of three neural architectures — **Fully Connected (FC), vanilla RNN, and LSTM** — on a *conditional source separation* task: extract one clean sinusoid from a noisy sum of four, given a one-hot selector vector.

### 1.2 Context
The lecturer's pedagogical thesis (HOMEWORK_BRIEF § 2) is that recurrent networks specialize by frequency regime: vanilla RNN is biased toward short-term/high-frequency structure (vanishing-gradient cost on long context) while LSTM, via its gating, retains long-term dependencies and should excel on low-frequency components. FC is a non-temporal baseline. The deliverable is not a working model — it is **evidence for, against, or refining this thesis**.

### 1.3 Problem statement
Given:
- A noisy summed signal $\tilde{S}(t) = \sum_{i=1}^{4} \tilde{s}_i(t)$ over 10 s at 1 kHz.
- A 10-sample window $W_{\text{noisy}}$ extracted from $\tilde{S}$ at a random start index.
- A one-hot selector $C \in \{0,1\}^4$ identifying one of the four base sinusoids.

Produce: a 10-sample prediction $\hat{W}_{\text{clean}}$ approximating the corresponding window of the **clean** selected sinusoid $s_k$, where $k = \arg\max(C)$.

### 1.4 Why this project matters
- Demonstrates command of the full SDLC under a strict engineering ruleset (`SOFTWARE_PROJECT_GUIDELINES.md`).
- Builds intuition for *when* gated recurrence pays for itself versus a memoryless baseline.
- Yields a reusable scaffold (uv-managed, SDK-first, TDD, CI-clean) for future deep-learning studies.

---

## 2. Target Audience

| Audience | Primary need |
| --- | --- |
| Course lecturer (Dr. Yoram Segal) | Verify thesis empirically; assess engineering hygiene per § 16 of the guidelines. |
| Future students of the course | Reference implementation, plus a worked research narrative they can read, run, and extend. |
| The author (ilyalaz01) | Reusable template for subsequent ML projects under the same ruleset. |

---

## 3. Goals & Success Criteria

### 3.1 Goals (in priority order, mirroring the lecturer's grading)
1. **Comparative analysis.** Produce evidence-based answers to: *Which architecture wins on each frequency regime, and why? Where does each fail?*
2. **Documentation depth.** Every non-trivial decision is recorded in its own ADR (`docs/adr/`) and every experiment in its own file (`docs/experiments/`). The top-level `README.md` is a research narrative with figures.
3. **Architectural understanding demonstrated in writing** — gating, vanishing gradients, memorylessness — referenced from numerical results.
4. **Engineering hygiene.** Full compliance with `SOFTWARE_PROJECT_GUIDELINES.md`: uv-only, ≤ 150 LOC per file, SDK-first, ≥ 85% coverage, zero `ruff` violations.

### 3.2 Measurable acceptance criteria
The project ships when ALL of the following hold:

| ID | Criterion | Verification |
| --- | --- | --- |
| AC-1 | All three architectures (FC, RNN, LSTM) train end-to-end on the corpus and produce per-frequency MSE on the held-out test split. | `notebooks/results.ipynb` reproduces the metric tables on a single `uv run` invocation. |
| AC-2 | A per-frequency MSE comparison table (4 frequencies × 3 models) is rendered with confidence intervals over ≥ 3 seeds. | Table present in `README.md` and in the analysis notebook. |
| AC-3 | The lecturer's thesis is explicitly evaluated: one statement per frequency regime saying whether RNN/LSTM bias is observed. | Section in `README.md` titled *"Thesis evaluation"*. |
| AC-4 | Test coverage ≥ 85%, `ruff check` passes with zero violations, all files ≤ 150 LOC. | CI script `scripts/check.sh` exits 0. |
| AC-5 | Every non-trivial design decision has a dedicated ADR. Every experiment run has a file in `docs/experiments/`. | `docs/adr/` and `docs/experiments/` directories populated; cross-referenced from PLAN.md and README.md. |
| AC-6 | `.env-example`, `pyproject.toml`, `uv.lock`, `.gitignore` all present and correct. | File presence + `uv sync` succeeds on a clean clone. |
| AC-7 | Public classes follow the Building Block docstring format (Input / Output / Setup) per § 15.1 of the guidelines. | Code review + automated docstring check (planned ADR). |

### 3.3 Non-goals (out of scope for v1.00)
- Non-sinusoidal sources (chirps, square waves, real audio).
- More than 4 sinusoidal components.
- Variable-length context windows. The window is fixed at 10 samples per HOMEWORK_BRIEF § 4.2.
- Generative or unsupervised approaches (autoencoders, ICA, NMF).
- Real-time inference, deployment, or productionization.
- Hyperparameter tuning beyond a documented sensitivity analysis on a small grid.

---

## 4. Functional Requirements

| ID | Requirement | Detail location |
| --- | --- | --- |
| FR-1 | Generate the 10-vector signal corpus (4 clean sines, 4 noisy sines, clean sum, noisy sum). | `PRD_signal_generation.md` |
| FR-2 | Sample training/validation/test windows with the conditional `[C, W_noisy] → W_clean_selected` contract. | `PRD_dataset_construction.md` |
| FR-3 | Implement FC, RNN, LSTM models behind a uniform interface, with selector-broadcast input scheme for RNN/LSTM (`[x_t, C]` per step). | `PRD_models.md` |
| FR-4 | Train each model with MSE loss; log losses, save checkpoints, support resume. | `PRD_training_evaluation.md` |
| FR-5 | Evaluate each model with overall MSE, per-frequency MSE, and qualitative reconstructions. | `PRD_training_evaluation.md` |
| FR-6 | Expose all of the above as a single SDK class consumed by CLI and notebooks. | `PLAN.md` (component diagram) + future ADR. |
| FR-7 | Persist all configuration in `config/*.json` (versioned) and secrets in `.env`. | `SOFTWARE_PROJECT_GUIDELINES.md` § 6. |
| FR-8 | Produce a research notebook with all figures referenced from `README.md`. | `notebooks/results.ipynb` |

---

## 5. Non-Functional Requirements

| ID | Requirement | Threshold / detail |
| --- | --- | --- |
| NFR-1 | Reproducibility | Fixed seeds; `uv sync` + `uv run scripts/train.py --config config/setup.json` recreates published numbers within the seed-CI. |
| NFR-2 | Device-agnostic determinism | The reference run path is reproducible on either CPU or CUDA. The device is read from configuration (`cuda` / `cpu` / `auto`). Determinism is enforced via fixed seeds and `torch.use_deterministic_algorithms` where supported by the chosen backend. |
| NFR-3 | Local-only | No external API calls. `ApiGatekeeper` is vacuously satisfied (planned ADR ADR-001). |
| NFR-4 | Linter cleanliness | `ruff check` returns zero violations. |
| NFR-5 | Test coverage | `pytest --cov` reports ≥ 85% statement and branch coverage. |
| NFR-6 | File size | Every Python file is ≤ 150 LOC (blank/comment lines excluded). |
| NFR-7 | Documentation density | Every non-trivial decision has an ADR; every experiment has a file. |
| NFR-8 | Package management | Only `uv` is used. No `pip`, `venv`, `virtualenv`, or `python -m`. |
| NFR-9 | Performance reporting | Training wall-clock time is measured per (model, device) pair and reported in `docs/experiments/`. The reference configuration completes in a single uninterrupted run; specific timings are populated after the first benchmark, not asserted up-front. |
| NFR-10 | Versioning | Code and config start at version `1.00`; bump on meaningful changes. |

---

## 6. Assumptions

- PyTorch is the modeling framework (planned ADR ADR-002).
- Sampling rate, signal duration, sinusoid count, and window size are fixed by the brief and not negotiable in v1.00.
- The selector $C$ is broadcast onto every timestep for RNN/LSTM (planned ADR ADR-003); HOMEWORK_BRIEF § 5.2 already adopts this.
- The runtime is **device-agnostic from day one**. The device (`cuda` / `cpu` / `auto`) is read from configuration. The reference device is selected after measurement, not before (see § 13.5 and the planned `ADR-006-reference-device.md`).
- The course environment trusts published `uv.lock`; a clean clone resolves to identical wheels.

---

## 7. Dependencies

### 7.1 External libraries (final list pinned in `pyproject.toml` once chosen)
- `torch` — model definition and training.
- `numpy` — signal generation, vectorized math.
- `matplotlib` (and possibly `seaborn`) — figures in the analysis notebook.
- `pytest`, `pytest-cov` — TDD harness.
- `ruff` — linting.
- `jupyterlab` (dev only) — notebook authoring.

### 7.2 Toolchain
- `uv` ≥ 0.11 as the single package manager and task runner.
- Python ≥ 3.12 (already pinned in `pyproject.toml`).

### 7.3 Course documents
- `HOMEWORK_BRIEF.md` — locked task specification.
- `SOFTWARE_PROJECT_GUIDELINES.md` — engineering ruleset.

---

## 8. Constraints

- Window size **must** be exactly 10 samples (HOMEWORK_BRIEF § 4.2).
- RNN/LSTM input feature size per step **must** be 5 (1 sample + 4 selector dims) to keep the three architectures comparable (HOMEWORK_BRIEF § 5.2).
- All business logic **must** be reachable through a single SDK layer (guidelines § 3.1). CLI and notebooks contain no logic.
- No hard-coded values (guidelines § 6.2).
- No secrets in source (guidelines § 6.4).

---

## 9. User Stories

- **U-1.** *As the course lecturer*, I want to clone the repository, run `uv sync && uv run scripts/train.py`, and see the published metrics reproduced, so I can verify the project actually trains.
- **U-2.** *As the course lecturer*, I want to read `README.md` end-to-end and understand the thesis, the experimental setup, and the conclusions without opening a single source file.
- **U-3.** *As a future student*, I want to find every design decision in its own ADR, so I can disagree with one and re-run that one experiment.
- **U-4.** *As the author*, I want to extend the corpus to 5 sinusoids in a future iteration without rewriting the dataset module.
- **U-5.** *As the author*, I want a one-line CLI to retrain any single model, so I can iterate on hyperparameters quickly.

---

## 10. Timeline & Milestones

The current date is **2026-04-30**. The working window is **7 days**. Detailed task tracking lives in `docs/TODO.md`.

| Milestone | Definition of Done | Target |
| --- | --- | --- |
| M0 — Documentation locked | PRD.md, PLAN.md, four dedicated PRDs, TODO.md drafted and approved. Initial ADRs (ADR-001 through ADR-005, covering ApiGatekeeper N/A, framework choice, selector-broadcast scheme, dataset-size, and noise distribution) written. | Day 1 |
| M1 + M2 — Skeleton + signal + dataset | `src/` package, SDK stub, config loader, version module, CI script. Signal-generation and dataset-construction modules implemented under TDD with ≥ 85% local coverage. | Day 2 |
| M3 + M4 — Models + training pipeline | FC / RNN / LSTM implemented behind a uniform interface (selector-broadcast scheme verified by tests). Full training loop with checkpointing, logging, device-agnostic execution, and the sensitivity-grid runner in place. CPU-vs-GPU benchmark on the smoke config recorded as `EXP-001`, feeding `ADR-006-reference-device.md`. | Day 3–4 |
| M5 — Experiments | Thesis-evaluation experiments run on the chosen reference device. Results saved under `results/`, one file per run in `docs/experiments/`. | Day 5 |
| M6 — Narrative | `README.md` rewritten as a research narrative with figures. Analysis notebook polished. Final per-frequency MSE table populated. | Day 6 |
| M7 — Buffer | Final compliance check (ruff, coverage, file-size, docstring format), polish pass on `README.md`, ADR cross-references audited. | Day 7 |

Slippage on M0 (Day 1) remains unacceptable — the guidelines mandate docs-before-code. Slippage on M5/M6 may consume the M7 buffer.

---

## 11. Risks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| The thesis fails to manifest cleanly (e.g. LSTM also wins on high frequencies). | Medium | Low | The grade rewards *honest analysis*, not confirmation. Document the negative result; this is in fact more interesting. |
| 150-LOC limit forces awkward splits in model code. | Medium | Low | Plan the split up front in `PRD_models.md` (one file per architecture, plus a shared base). |
| Phase noise spanning $0$ to $2\pi$ erases information faster than expected. | Medium | Medium | Sensitivity sweep on $\beta$ documented as an experiment. |
| Coverage metric pulled down by notebook code. | Low | Low | `coverage.run.omit` excludes notebooks and GUI per guidelines § 5.2. |
| Selector-broadcast scheme is suboptimal vs. hidden-state init. | Medium | Low | Documented as a planned ablation; out of scope for v1.00. |

---

## 12. Out of Scope (explicit)

- Real audio data.
- Transformer or attention-based baselines.
- Multi-window outputs (we predict only the 10 selected samples).
- Online/streaming training.
- Multi-GPU or distributed training.

---

## 13. Resolutions of Open Questions (locked 2026-04-30)

Each item below is resolved and will be promoted to its own ADR before the M0 deadline.

1. **Dataset size — LOCKED.** 30 000 train / 3 750 val / 3 750 test windows (80 / 10 / 10 split). → `ADR-004-dataset-size.md`.
2. **Noise distribution — LOCKED.** Gaussian for both $\epsilon_A$ and $\epsilon_\phi$; phase wrapped into $[0, 2\pi)$. → `ADR-005-noise-distribution.md`.
3. **Number of seeds — LOCKED.** 3 seeds per (model, frequency-regime) cell; mean and 95% CI reported. → ADR-007 (planned alongside experiment design).
4. **Frequency set — LOCKED.** 2 / 10 / 50 / 200 Hz as in HOMEWORK_BRIEF § 8. → ADR-008 (planned alongside `PRD_signal_generation.md`).
5. **CPU vs GPU — RESOLVED, MEASUREMENT-DRIVEN.** The codebase is device-agnostic from day one (`cuda` / `cpu` / `auto` read from config). The reference device is chosen by ADR **after** the EXP-001 smoke benchmark in M3+M4: GPU is selected if it is ≥ 10× faster than CPU on the smoke configuration; otherwise CPU. The decision lands in `ADR-006-reference-device.md`, written *after* measurement.

The seed/frequency ADRs (ADR-007, ADR-008) may be deferred to M0 day end because they do not gate scaffolding work; ADR-001 through ADR-006 are M0-blocking.

---

## 14. References

- `HOMEWORK_BRIEF.md` — task specification.
- `SOFTWARE_PROJECT_GUIDELINES.md` — engineering ruleset.
- `docs/PLAN.md` — architecture (to be drafted next).
- `docs/PRD_signal_generation.md`, `docs/PRD_dataset_construction.md`, `docs/PRD_models.md`, `docs/PRD_training_evaluation.md` — mechanism-level PRDs (to follow).
- `docs/adr/` — architectural decisions (incremental).
- `docs/experiments/` — experiment logs (incremental).

# TODO — Signal Source Extraction Study

> **Last updated:** 2026-05-01
> **Owner:** Student (all tasks unless noted).
> **Milestone status:** M0 in progress. M1–M7 not started.
>
> This is the live task board. Update checkbox state as each item completes. Add sub-items inline when a task spawns unforeseen work. Do not delete completed items — history is useful for the PROMPTS log and grader.

---

## M0 — Documentation lockdown (in progress)

All code-facing files are blocked until every item in M0 is checked and the user has reviewed and approved each document.

### Core PRDs & PLAN

- [x] `docs/PRD.md` v1.01 — drafted, reviewed, approved
- [x] `docs/PLAN.md` v1.01 — drafted, reviewed, approved
- [x] `docs/PRD_signal_generation.md` v1.01 — drafted, reviewed, approved
- [x] `docs/PRD_dataset_construction.md` v1.01 — drafted, reviewed, approved
- [x] `docs/PRD_models.md` v1.01 — drafted, reviewed, approved
- [x] `docs/PRD_training_evaluation.md` v1.01 — drafted, reviewed, approved

### Mandatory companion docs (created this session)

- [x] `docs/TODO.md` — this file
- [x] `docs/PROMPTS.md` — prompt engineering log

### M0 ADRs (upfront — per PLAN.md § 13.1)

- [x] `docs/adr/ADR-003-selector-broadcast.md` — written alongside PRD_models.md
- [x] `docs/adr/ADR-016-random-sampling-stationary.md` — written alongside PRD_dataset_construction.md v1.01
- [x] `docs/adr/ADR-017-dataloader-shuffling.md` — written alongside PRD_dataset_construction.md v1.01
- [x] `docs/adr/ADR-001-gatekeeper-na.md`
- [x] `docs/adr/ADR-002-pytorch.md`
- [x] `docs/adr/ADR-004-dataset-size.md`
- [x] `docs/adr/ADR-005-noise-distribution.md`
- [x] `docs/adr/ADR-009-component-decomposition.md`
- [x] `docs/adr/ADR-012-gatekeeper-absent.md`
- [x] `docs/adr/ADR-014-results-layout.md`
- [x] `docs/adr/ADR-015-base-phase-distribution.md`

### PLAN v1.02 sweep ✓ complete (2026-05-01)

- [x] Remove stale `split_strategy` field from config schema in PLAN.md § 9
- [x] Fold ADR-016 / ADR-017 references into PLAN.md § 13 (all M0 ADRs now listed as written)
- [x] Tighten model package LOC ceiling in PLAN.md § 8 (310 → 240; `__init__` 10→30, `fc/rnn/lstm` 80→50 each)
- [x] Tighten dataset.py LOC ceiling in PLAN.md § 8 (~120 → ~100)
- [x] Write deferred M0 ADRs: 001, 002, 004, 005, 009, 012, 014, 015
- [x] Add § 11.5 Device (device-agnostic NFR) to PLAN.md
- [x] Add § 15 cross-references to companion PRDs in PLAN.md
- [x] Fix "planned" markers across all four dedicated PRDs
- [x] Add § 15 v1.02 follow-up checklist to PRD_dataset_construction.md

---

## M1 — Skeleton green

*Prerequisite: all M0 items checked and the user has said "proceed to M1".*

- [ ] `uv init --package` if `pyproject.toml` absent; verify `uv.lock` committed
- [ ] Add core dependencies: `torch`, `numpy`, `pytest`, `pytest-cov`, `ruff`
- [ ] `.gitignore` (include `.env`, `*.pem`, `results/`, `__pycache__/`, `.venv/`)
- [ ] `.env-example` with placeholder values (no secrets in code)
- [ ] `src/signal_extraction_rnn_lstm/` scaffold:
  - [ ] `__init__.py` (exports SDK, `__version__`)
  - [ ] `constants.py`
  - [ ] `sdk/__init__.py`, `sdk/sdk.py` (stub — signatures only, no logic)
  - [ ] `services/__init__.py`
  - [ ] `shared/__init__.py`, `shared/version.py`, `shared/config.py`, `shared/seeding.py`, `shared/device.py`
- [ ] `tests/unit/`, `tests/integration/`, `tests/conftest.py` scaffolded (empty)
- [ ] `config/setup.json` v1.00, `config/rate_limits.json` v1.00 (vestigial schema)
- [ ] `scripts/check.sh` (runs `uv run ruff check src/ tests/` then `uv run pytest --cov=src --cov-fail-under=85`)
- [ ] `uv run scripts/check.sh` passes on the empty scaffold (no violations, coverage trivially ≥ 85% on near-empty files)

---

## M2 — Signal & dataset

*Prerequisite: M1 complete.*

- [ ] RED: write `tests/unit/test_signal_gen.py` (all T-SG-* from PRD_signal_generation.md § 9)
- [ ] GREEN: implement `services/signal_gen.py`
- [ ] REFACTOR: `ruff check` clean; file ≤ 150 LOC
- [ ] RED: write `tests/unit/test_dataset.py` (all T-DS-* from PRD_dataset_construction.md § 9)
- [ ] GREEN: implement `services/dataset.py`
- [ ] REFACTOR: `ruff check` clean; file ≤ 100 LOC (internal target)
- [ ] Coverage ≥ 95% on both modules
- [ ] Integration smoke: `SDK.generate_corpus()` → `SDK.build_dataset()` → `len(splits.train)` succeeds

---

## M3 — Models

*Prerequisite: M2 complete.*

- [ ] RED: write `tests/unit/test_models/test_base.py` (T-MD-01, T-MD-02, T-MD-03)
- [ ] RED: write `tests/unit/test_models/test_fc.py` (T-MD-04, T-MD-07, T-MD-10, T-MD-13, T-MD-16)
- [ ] RED: write `tests/unit/test_models/test_rnn.py` (T-MD-05, T-MD-08, T-MD-11, T-MD-14, T-MD-17)
- [ ] RED: write `tests/unit/test_models/test_lstm.py` (T-MD-06, T-MD-09, T-MD-12, T-MD-15, T-MD-18)
- [ ] RED: write `tests/unit/test_models/test_registry.py` (T-MD-19, T-MD-20)
- [ ] GREEN: implement `services/models/base.py`, `fc.py`, `rnn.py`, `lstm.py`, `__init__.py`
- [ ] REFACTOR: `ruff check` clean; each file ≤ its PLAN.md § 8 per-file ceiling
- [ ] All T-MD-01 through T-MD-20 pass
- [ ] Coverage ≥ 90% on `services/models/`
- [ ] Integration smoke: `tests/integration/test_models_smoke.py` (AC-MD-6)

---

## M4 — Training pipeline

*Prerequisite: M3 complete.*

- [ ] RED: write `tests/unit/test_training.py` (T-TR-* from PRD_training_evaluation.md)
- [ ] RED: write `tests/unit/test_evaluation.py` (T-EV-* from PRD_training_evaluation.md)
- [ ] GREEN: implement `services/training.py`
- [ ] GREEN: implement `services/evaluation.py`
- [ ] REFACTOR: `ruff check` clean; training.py ≤ 140 LOC, evaluation.py ≤ 130 LOC
- [ ] Wire `SDK.train()`, `SDK.evaluate()`, `SDK.run_experiment()` to services
- [ ] `scripts/train.py` and `scripts/run_experiment.py` thin wrappers
- [ ] Integration smoke: 1 epoch of each model kind on a tiny corpus, metrics returned, checkpoint saved
- [ ] `tests/integration/test_reproducibility.py`: same seed → identical loss values
- [ ] Coverage ≥ 85% overall

---

## M5 — Experiments

*Prerequisite: M4 complete.*

- [ ] `EXP-001`: three-way baseline (FC / RNN / LSTM) on full dataset, 30 epochs, default config
  - [ ] Document in `docs/experiments/EXP-001-baseline.md`
  - [ ] Commit per-frequency MSE table to `results/`
- [ ] `EXP-002` through `EXP-004`: sensitivity sweeps (noise α, noise β, dataset size) — write ADR-007 first
- [ ] `EXP-005`: bidirectional ablation (per § 13.1 of PRD_models.md)
- [ ] `EXP-006`: hidden-state init scheme ablation (deferred from PRD_models.md § 13.3)
- [ ] `EXP-007`: sequence-to-sequence head ablation
- [ ] `EXP-008`: parameter-matched RNN (hidden ≈ 128) vs LSTM (hidden 64)
- [ ] Write deferred ADRs triggered by EXP-001: ADR-006 (device), ADR-007 (seeds per cell)
- [ ] `notebooks/results.ipynb`: loss curves, per-frequency MSE, FC-vs-RNN-vs-LSTM comparison plots
- [ ] Figures committed to `assets/figures/`
- [ ] `AC-DS-9` (t_0-histogram figure) and `AC-DS-5` (dataset example figure) delivered here

---

## M6 — Narrative

*Prerequisite: M5 complete.*

- [ ] `README.md` as full research narrative + user manual (per guidelines § 1.1 and HOMEWORK_BRIEF.md § 10.2)
  - [ ] Installation instructions (uv-based)
  - [ ] Screenshots of signals, loss curves, error plots, architecture diagrams
  - [ ] Comparative analysis section: when each architecture wins, why, what breaks
  - [ ] Failed experiments included
  - [ ] Framing sentence from ADR-016 reproduced verbatim (AC-DS-9 requirement)
- [ ] `notebooks/results.ipynb` polished and narrative-complete
- [ ] All acceptance criteria (AC-*) verified across all PRDs

---

## M7 — Buffer / final compliance

*Prerequisite: M6 complete.*

- [ ] File-size audit: `find src/ tests/ -name "*.py" | xargs wc -l` — flag any file approaching 130 LOC
- [ ] `uv run ruff check src/ tests/` — zero violations
- [ ] `uv run pytest --cov=src --cov-fail-under=85` — green
- [ ] `uv.lock` and `pyproject.toml` committed and current
- [ ] `.env-example` has no live secrets
- [ ] Git history clean (meaningful commit messages, no debug commits)
- [ ] Final `scripts/check.sh` run on a clean clone

---

## Deferred / conditional items

These are not in the critical path but are tracked so they are not forgotten:

| Item | Trigger | ADR / Note |
| --- | --- | --- |
| Input normalization | EXP-001 shows optimizer instability | Would become ADR-018 |
| GitHub Actions CI | User requests remote CI | ADR-010 |
| Config override `--override key=value` | CLI usage demands it | ADR-013 |
| Logging framework upgrade | stdout-plus-file proves insufficient | ADR-011 |

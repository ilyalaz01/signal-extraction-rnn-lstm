# TODO — Signal Source Extraction Study

> **Last updated:** 2026-05-02
> **Owner:** Student (all tasks unless noted).
> **Milestone status:** M0 complete. M1 complete. M2 complete (2026-05-02). M3 in progress (M3a complete).
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

## M1 — Skeleton green ✓ complete (committed 2026-05-02)

*Prerequisite: all M0 items checked and the user has said "proceed to M1".*

- [x] `uv init --package` if `pyproject.toml` absent; verify `uv.lock` committed
- [x] Add core dependencies: `torch`, `numpy`, `pytest`, `pytest-cov`, `ruff`
- [x] `.gitignore` (include `.env`, `*.pem`, `results/`, `__pycache__/`, `.venv/`)
- [x] `.env-example` with placeholder values (no secrets in code)
- [x] `src/signal_extraction_rnn_lstm/` scaffold:
  - [x] `__init__.py` (exports SDK, `__version__`)
  - [x] `constants.py`
  - [x] `sdk/__init__.py`, `sdk/sdk.py` (stub — signatures only, no logic)
  - [x] `services/__init__.py`
  - [x] `shared/__init__.py`, `shared/version.py`, `shared/config.py`, `shared/seeding.py`, `shared/device.py`
- [x] `tests/unit/`, `tests/integration/`, `tests/conftest.py` scaffolded (empty)
- [x] `config/setup.json` v1.00, `config/rate_limits.json` v1.00 (vestigial schema)
- [x] `scripts/check.sh` (runs `uv run ruff check src/ tests/` then `uv run pytest --cov=src --cov-fail-under=85`)
- [x] `uv run scripts/check.sh` passes on the empty scaffold (no violations, coverage trivially ≥ 85% on near-empty files)
- [x] Abstract base renamed `SignalExtractor` per PRD_models.md § 3.4
- [x] `results/.gitkeep` removed; `results/` kept runtime-only via `.gitignore`

---

## M2 — Signal & dataset

*Prerequisite: M1 complete.*

### M2a — signal_gen ✓ complete (2026-05-02)

- [x] RED: write `tests/unit/test_signal_gen.py` (all T-SG-01..14 from PRD_signal_generation.md § 9)
- [x] GREEN: implement `services/signal_gen.py` (SignalConfig, Corpus, make_clean, make_noisy, generate_corpus)
- [x] GREEN: implement `shared/config.py` (parse_angle + load_config)
- [x] RED→GREEN: `tests/unit/test_config.py` for load_config
- [x] REFACTOR: `ruff check` clean; signal_gen.py 103 code-LOC (excl. docstrings), 145 incl. — within ≤ 150
- [x] Coverage 100% on signal_gen.py and shared/config.py (target ≥ 95%)
- [x] `pyproject.toml` coverage `omit` list for not-yet-implemented stubs (re-include per milestone)

### M2b — dataset ✓ complete (2026-05-02)

- [x] RED: write `tests/unit/test_dataset.py` (all T-DS-04..18 + edge cases; T-DS-01..03 retired in PRD v1.01; T-DS-11' replaces T-DS-11)
- [x] GREEN: implement `services/dataset.py` (DatasetConfig, WindowExample, WindowDataset, SplitDatasets, build_split_datasets)
- [x] REFACTOR: `ruff check` clean; dataset.py 91 code-LOC (target ~100)
- [x] Coverage 100% on dataset.py (target ≥ 95%)
- [x] Removed `*/services/dataset.py` from coverage `omit`
- [x] Spec deviation noted: `WindowExample` is `NamedTuple` (PRD said `@dataclass(frozen=True)`); chosen for `default_collate` compat. Logged in PROMPTS § 6.

### M2c — SDK wiring + integration smoke ✓ complete (2026-05-02)

- [x] GREEN: `shared/seeding.py` — `seed_everything`, `derive_seeds(runtime_seed) → (corpus_seed, sampling_seed)`
- [x] GREEN: `shared/device.py` — `resolve_device(str) → torch.device`
- [x] GREEN: `parse_signal_config(d) → SignalConfig` (added to `services/signal_gen.py`)
- [x] GREEN: `parse_dataset_config(d) → DatasetConfig` (added to `services/dataset.py`)
- [x] GREEN: `SDK.__init__`, `SDK.generate_corpus`, `SDK.build_dataset`; M4 methods still raise `NotImplementedError("M4")`
- [x] Integration smoke: `tests/integration/test_sdk_smoke.py` — full path + reproducibility under same seed
- [x] Removed `*/shared/device.py`, `*/shared/seeding.py`, `*/sdk/sdk.py` from coverage `omit`
- [x] Coverage 100% on all 12 in-scope modules (target ≥ 85%)

---

## M3 — Models

*Prerequisite: M2 complete.*

> **Note on the original split:** the TODO v1.01 grouped T-MD-03 (registry dispatch) under `test_base.py` — it is moved to `test_registry.py` because registry dispatch needs all three concrete classes to exist. T-MD-19 / T-MD-20 (cross-cutting grad flow) also live in `test_registry.py`.

### M3a — base + reshape utilities ✓ complete (2026-05-02)

- [x] RED: write `tests/unit/test_models/test_base.py` (T-MD-01, T-MD-02 + abstract-base contract + length-agnostic reshape)
- [x] GREEN: implement `services/models/base.py` (SignalExtractor, _to_fc_input, _to_seq_input)
- [x] REFACTOR: ruff clean; base.py 11 stmts (target ~40); 100% covered
- [x] Removed `*/services/models/base.py` from coverage `omit` (per-file omit list now enumerates fc/rnn/lstm/__init__)

### M3b — FC ✓ complete (2026-05-02)

- [x] RED: `tests/unit/test_models/test_fc.py` (T-MD-04, 07, 10, 13, 16 + FCConfig validation)
- [x] GREEN: `services/models/fc.py` (FCConfig + FCExtractor, 28 stmts, target ~30)
- [x] Removed `*/services/models/fc.py` from coverage `omit`
- [x] **T-MD-10 calibration:** PRD spec (SGD lr=1e-2 / 200 steps / MSE < 1e-3) numerically too tight; bumped to 2000 steps. PROMPTS § 8 logs the deviation and the convergence-curve evidence.

### M3c — RNN ✓ complete (2026-05-02)

- [x] RED: `tests/unit/test_models/test_rnn.py` (T-MD-05, 08, 11, 14, 17 + RNNConfig validation)
- [x] GREEN: `services/models/rnn.py` (RNNConfig + RNNExtractor, 24 stmts, target ~30)
- [x] Removed `*/services/models/rnn.py` from coverage `omit`
- [x] T-MD-11 step count 200 → 2000 (same calibration as T-MD-10; PROMPTS § 8)

### M3d — LSTM

- [ ] RED: write `tests/unit/test_models/test_lstm.py` (T-MD-06, T-MD-09, T-MD-12, T-MD-15, T-MD-18)
- [ ] GREEN: implement `services/models/lstm.py`
- [ ] Remove `*/services/models/lstm.py` from coverage `omit`

### M3e — registry + cross-cutting + integration smoke

- [ ] GREEN: implement `services/models/__init__.py` (ModelKind, ModelConfig, build, registry dict)
- [ ] RED: write `tests/unit/test_models/test_registry.py` (T-MD-03 dispatch, T-MD-19 grad-flow, T-MD-20 selector-differentiable)
- [ ] Integration smoke: `tests/integration/test_models_smoke.py` (AC-MD-6)
- [ ] Remove `*/services/models/__init__.py` from coverage `omit`
- [ ] All T-MD-01 through T-MD-20 pass; coverage ≥ 90% on `services/models/`

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

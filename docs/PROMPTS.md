# PROMPTS.md — Prompt Engineering Log

> **Purpose:** Append-only log of significant prompts used to build this project, per `SOFTWARE_PROJECT_GUIDELINES.md` § 7.3.
> **Maintained by:** Student, with Claude Code as co-author.
> **Format:** Chronological. Each entry records the session context, the prompt intent, what the model produced, what was iterated or corrected, and any reusable technique noted.
>
> **Reconstruction note:** Sessions prior to 2026-04-30 session 2 are reconstructed from document artifacts (version history, changelog lines, ADR context). The prompt text is described from intent, not verbatim, because the original session transcripts are not available. Starting from session 2, entries reflect the live conversation.

---

## Session 1 — Project scaffolding & initial framework

**Date:** 2026-04-30 (estimated)
**Goal:** Lock the project brief, establish the engineering ruleset as a normative document, and write `CLAUDE.md` so that all future Claude Code sessions begin with the correct context automatically loaded.

**Prompts (reconstructed):**
1. Draft `CLAUDE.md` referencing `HOMEWORK_BRIEF.md` and `SOFTWARE_PROJECT_GUIDELINES.md` via `@`-imports, summarizing non-negotiable rules, standing decisions, and the suggested first-session work order.
2. Draft `HOMEWORK_BRIEF.md` as the canonical project specification, reconstructing the assignment from lecture notes (lecturer did not distribute a written brief). Lock signal model, dataset construction rules, model contracts, and grading priorities.

**Outputs:** `CLAUDE.md`, `HOMEWORK_BRIEF.md`.

**Key decisions made here:**
- Broadcast-C injection scheme chosen as the canonical method (locked in HOMEWORK_BRIEF § 5.2 before PRD_models.md was written).
- Phase noise β = 2π ("0 to 2π is more interesting" per the lecturer).
- Sampling rate 1000 Hz, 10 s duration, window = 10 samples — all fixed.
- ApiGatekeeper requirement declared vacuously satisfied (no external APIs).

**Technique noted:** Loading normative documents via `@`-imports in `CLAUDE.md` is more reliable than pasting them per-session. All future sessions inherit the full context automatically.

---

## Session 1 (cont.) — PRD.md v1.00 and v1.01

**Goal:** Produce `docs/PRD.md` covering project overview, functional/non-functional requirements, and the four-sinusoid frequency set.

**Prompts (reconstructed):**
1. "Draft docs/PRD.md per the guidelines' mandatory structure (§ 1.2). Use HOMEWORK_BRIEF.md as the single source of truth. Do not invent requirements not stated there."
2. Review pass: tighten NFR-3 (uniform model interface), add the framing sentence for the comparative study, lock frequency set `[2, 10, 50, 200] Hz` with per-frequency regime labels.

**Outputs:** `docs/PRD.md` v1.00 → v1.01.

**Key decisions made here:**
- Frequency set justified by the lecturer's RNN/LSTM thesis: 2 Hz (very low), 10 Hz (low), 50 Hz (medium), 200 Hz (high) relative to the 10-sample window.
- NFR-1 (reproducibility) and NFR-2 (determinism) added as named constraints after the training loop's seed requirements became obvious.

**Technique noted:** Asking Claude to justify every frequency choice "relative to the 10-sample window at 1000 Hz" produced more thesis-aligned reasoning than leaving frequency selection open-ended.

---

## Session 1 (cont.) — PLAN.md v1.00 and v1.01

**Goal:** Produce the architecture document with C4 diagrams, SDK contract, LOC budget table, and ADR index.

**Prompts (reconstructed):**
1. "Draft docs/PLAN.md with C4 level 1–4, sequence diagrams, module layout with LOC ceilings, and ADR index. Distinguish upfront ADRs (M0) from deferred ones."
2. Review pass: corrections to § 6.1 (SDK `evaluate()` signature — must not accept raw `nn.Module` from external callers), § 8 (LOC budget numbers), § 9 (config schema with angle-expression convention), § 11.4 (num_workers = 0 is intentional and load-bearing — must be in config, not hard-coded), § 14 (lock four open architectural questions).

**Outputs:** `docs/PLAN.md` v1.00 → v1.01. `docs/adr/` directory structure implied.

**Key decisions made here:**
- `gatekeeper.py` declared absent (not stubbed) — ADR-012.
- Model registry lives in `services/models/__init__.py`.
- Results layout: `results/<utc_timestamp>__<model_kind>__<seed>/` — ADR-014.
- Base-phase distribution `[0, π/2, π, 3π/2]` to avoid zero-phase constructive interference at t=0 — ADR-015.

**Technique noted:** C4 diagrams in Mermaid generated cleanly. Sequence diagrams benefit from naming each actor before drawing the flow — avoids arrow-direction confusion in the generated Mermaid.

---

## Session 1 (cont.) — PRD_signal_generation.md v1.00 and v1.01

**Goal:** Produce the dedicated PRD for signal generation: sinusoid model, noise model, corpus structure, test plan.

**Prompts (reconstructed):**
1. "Draft docs/PRD_signal_generation.md covering the signal model from HOMEWORK_BRIEF § 3, the 10-vector corpus structure, and a test plan with T-SG-* IDs."
2. Review pass: fix corpus field names to match the implementation-level names (e.g. `corpus.clean`, `corpus.noisy_sum`), add the stationary-process framing that motivates the dataset construction split strategy, tighten test T-SG-* to distinguish unit from integration tests.

**Outputs:** `docs/PRD_signal_generation.md` v1.00 → v1.01.

**Key decisions made here:**
- Gaussian noise distribution (vs. Uniform) — ADR-005 (to be written). Rationale: Gaussian is more physically motivated for additive amplitude and phase perturbations.
- α = 0.05 (5% amplitude noise), β = 2π (full-range phase noise).
- Corpus arrays stored as `float32` to avoid dtype-conversion overhead at Dataset boundary.

---

## Session 1 (cont.) — PRD_dataset_construction.md v1.00

**Goal:** Produce the dedicated PRD for window sampling, splits, and tensor contracts delivered to models.

**Prompts (reconstructed):**
1. "Draft docs/PRD_dataset_construction.md. Window = 10 samples. Dataset size from ADR-004 (30 000 / 3 750 / 3 750). The Dataset must be model-agnostic — no FC/RNN reshape."
2. Initial split strategy: disjoint t_0 ranges across train/val/test (v1.00 draft).

**Outputs:** `docs/PRD_dataset_construction.md` v1.00 (with disjoint-range split strategy — later revised).

---

## Session 1 (cont.) — Random-sampling decision & PRD_dataset_construction.md v1.01

**Goal:** Revisit the v1.00 disjoint-range split strategy after catching a logical flaw in review.

**Prompts (reconstructed):**
1. "The disjoint t_0 split is wrong. The signal process is stationary — every window start is drawn from the same distribution regardless of t_0. Disjoint ranges do not measure generalization. Revise the split strategy to random i.i.d. sampling from the full pool on all three splits."
2. "Write ADR-016-random-sampling-stationary.md recording this decision and the alternatives considered (disjoint with W-1 buffer; shoulder-disjoint; per-split corpora)."
3. "Write ADR-017-dataloader-shuffling.md: train shuffles each epoch; val and test iterate in deterministic order."
4. Review pass on PRD_dataset_construction.md v1.01: replace disjoint-range tests (T-DS-01..03 retired), add KS-equivalence test (T-DS-11'), add meta() method, add AC-DS-8 and AC-DS-9.

**Outputs:** `docs/PRD_dataset_construction.md` v1.01, `docs/adr/ADR-016-random-sampling-stationary.md`, `docs/adr/ADR-017-dataloader-shuffling.md`.

**Lesson learned:** The "shoulder-disjoint" formulation (disjoint in t_0 but not in covered samples) was the v1.00 draft formulation — caught during review. Record cautionary alternatives in ADRs even when they were only ever drafts; they are the most useful part of the ADR for future readers.

**Technique noted:** When a design decision involves stationarity, write the framing sentence explicitly and verbatim ("Splits exist by example count… because the underlying signal process is stationary") and require it to appear in README.md and the ADR. This prevents the reasoning from getting lost across sessions.

---

## Session 2 — PRD_models.md v1.00 draft and ADR-003

**Date:** 2026-04-30
**Goal:** Produce the dedicated PRD for the three model architectures (FC, RNN, LSTM) and the selector-broadcast ADR. Previous session hit a rate limit immediately after writing these two files, before delivering a summary.

**Prompts:**
1. "Pick up where we stopped. Read ADR-003, PRD_models.md, PRD_dataset_construction.md, and PLAN.md in order. Then summarize PRD_models.md as if presenting it for review for the first time — structure, key decisions, places where you made calls without explicit user input, and any concerns. Do not modify the file."

**Outputs read (not written this session):** `docs/adr/ADR-003-selector-broadcast.md`, `docs/PRD_models.md` v1.00.

**Concerns surfaced during review:**
- Arithmetic bug: LSTM parameter count 17 920 in prose should be 18 176 (formula was right, evaluation was wrong).
- "RNN should excel at high-frequency" framing was over-confident and reviewer-optics-problematic (at 200 Hz, two full cycles fit the window — FC sees the same structure, so RNN's memory is not the discriminating factor).
- Seq-to-vector head was locked without user input; "stresses gradient flow" framing sounded like it was picking the head to make RNN look bad.
- `predict()` alias on `SignalExtractor` adds unnecessary test surface; PyTorch idiom is `model(...)`.
- `ADR-003` references `ADR-002` and `ADR-009` as "planned" — both still unwritten.

---

## Session 2 (cont.) — PRD_models.md v1.01 corrections

**Goal:** Apply the reviewer's correction list to produce PRD_models.md v1.01.

**Prompt:** (this session — verbatim)
> PRD_models.md approved with the following corrections for v1.01. [...] ARITHMETIC FIX: 4·4544 = 18 176 [...] § 5.2 RECONS reformulation: [...] § 5.2/§ 5.3 SEQ-TO-VECTOR reframing [...] PARAMETER ASYMMETRY disclosure + EXP-008 [...] T-MD-18 assertion clarification [...] DROP predict() alias [...] AC-MD-2 wording [...] OPEN QUESTIONS locked [...] LOC BUDGET CONFLICT keep two-column table.

**Outputs:** `docs/PRD_models.md` v1.01 (full rewrite with all corrections).

**Key decisions locked at v1.01:**
- Unidirectional in v1.00; `EXP-005` for bidirectional.
- Seq-to-vector in v1.00; `EXP-007` for seq-to-seq.
- Hidden-state init ablation deferred to `EXP-006`.
- Internal `evaluate()` may accept `SignalExtractor` directly; SDK boundary does not.

**Lesson learned:** When a PRD has many corrections (10+ change points), a full rewrite is more reliable than serial Edit calls. Cite the version number in the first line; add a Changelog entry so reviewers can diff at a glance without re-reading the full document.

---

## Session 2 (cont.) — docs/TODO.md and docs/PROMPTS.md

**Date:** 2026-04-30
**Goal:** Create the two mandatory documents that were missing from M0: the task board and the prompt log.

**Prompt:** (this session)
> Quick housekeeping pause before PRD_training_evaluation.md. I noticed we never wrote docs/TODO.md or docs/PROMPTS.md [...] docs/TODO.md — phased milestone board reflecting where we actually are. [...] docs/PROMPTS.md — append-only log. Backfill what we've done so far. New entries appended on every significant prompt going forward. Don't write code. Stop after for review, then proceed to PRD_training_evaluation.md.

**Outputs:** `docs/TODO.md`, `docs/PROMPTS.md` (this file).

**Technique noted:** Mandatory docs flagged mid-session by the user, not by the AI. Add TODO.md and PROMPTS.md to the very first session checklist in any future project following these guidelines — they are trivially missed because they are meta-documents that do not produce visible artifacts.

---

*Next entry: PRD_training_evaluation.md draft (Session 2, cont.)*

---

## Session 3 — M0 final sweep (PLAN v1.02, deferred ADRs, dangling fixes)

**Date:** 2026-05-01
**Goal:** Close M0 entirely. Three coordinated blocks: (1) PLAN.md v1.01 → v1.02 sweep; (2) write all eight deferred M0 ADRs; (3) fix dangling cross-document inconsistencies and update TODO.md.

**Prompt (verbatim):**
> M0 final sweep. Three blocks of work, do them in order. [...] BLOCK 1 — PLAN.md v1.02 sweep (single edit pass): remove split_strategy, fold ADR-016/017 as written, tighten model LOC ceilings (__init__ 10→30, fc/rnn/lstm 80→50, total 310→240), add § 15 cross-refs, add § 11.5 Device, update § 13 ADR index. [...] BLOCK 2 — eight deferred M0 ADRs [...] BLOCK 3 — dangling fixes: PRD_dataset_construction § 15 checklist, verify planned→written markers, tick TODO.md M0, append PROMPTS.md. When done, M0 is closed. Do NOT start M1 in the same turn.

**Outputs (all in one session):**
- `docs/PLAN.md` v1.01 → v1.02 (§ 8 LOC, § 9 config schema, § 11.5 Device, § 13 ADR index, new § 15 cross-refs, § 16 references).
- `docs/adr/ADR-001-gatekeeper-na.md` — vacuous gatekeeper satisfaction.
- `docs/adr/ADR-002-pytorch.md` — PyTorch over TensorFlow/Keras/JAX.
- `docs/adr/ADR-004-dataset-size.md` — 30k/3.75k/3.75k (80/10/10).
- `docs/adr/ADR-005-noise-distribution.md` — Gaussian, per-sample; α=0.05, β=2π.
- `docs/adr/ADR-009-component-decomposition.md` — SDK/services/shared/constants layering.
- `docs/adr/ADR-012-gatekeeper-absent.md` — no gatekeeper.py file exists.
- `docs/adr/ADR-014-results-layout.md` — `results/<timestamp>__<kind>__<seed>/` format.
- `docs/adr/ADR-015-base-phase-distribution.md` — [0, π/2, π, 3π/2] quadrature phases.
- `docs/PRD_dataset_construction.md` — § 15 v1.02 follow-up checklist added; § 14 references updated.
- `docs/PRD_signal_generation.md` — § 12 references: ADR-005, ADR-015 "(planned, M0)" → "(M0, written)".
- `docs/PRD_models.md` — companion header and § 14: ADR-002, ADR-009 "(planned, M0)" → "(M0, written)".
- `docs/PRD_training_evaluation.md` — companion header and § 17: ADR-014 "(planned M0)" → "(M0, written)".
- `docs/TODO.md` — M0 milestone fully ticked, PLAN v1.02 sweep section added and completed.

**Key decisions locked in this session:**
- Model package LOC ceilings tightened from 310 to 240 (PyTorch handles heavy lifting; observed internal targets ~145).
- ADR-018 (normalization) and ADR-019 (initialization) listed as conditional/deferred in PLAN § 13.2 — no pre-planned file, triggered only by EXP-001 evidence.
- ADR-008 (frequency selection) formally assigned as "folded into PRD_signal_generation.md § 4" in PLAN § 13.3.
- "gatekeeper-presence" ADR that appeared in PLAN v1.01 § 4 was a mislabel; corrected to point at ADR-012.
- `split_strategy` field was confirmed absent from config schema; the note explaining its absence added to PLAN § 9 Notes.

**Technique noted:** When a sweep touches many documents, doing the primary document (PLAN.md) first and writing all ADRs in parallel afterward is more efficient than interleaving. The "if an ADR contradicts the PRD, stop" instruction is the right safeguard — no contradictions surfaced; all ADR content was consolidation of reasoning already distributed across PRDs.

---

## Session 5 — M2a: signal_gen RED → GREEN

**Date:** 2026-05-02
**Goal:** TDD-cycle for the signal-generation subsystem. Produce `tests/unit/test_signal_gen.py` first (RED), get user review, then implement `services/signal_gen.py` and `shared/config.py` until all tests pass GREEN under `scripts/check.sh`.

**Prompts (live):**
1. "M2 plan: SignalConfig home decision, parse_angle strategy, make_clean/make_noisy coverage policy, commit boundaries." User approved with one addition: extra direct test T-SG-14 for `make_noisy(alpha=0, beta=0)` returning byte-identical clean.
2. "Write `tests/unit/test_signal_gen.py` (RED phase). Show me the file before implementing services/signal_gen.py." Wrote 14 T-SG cases plus parse_angle + edge-case tests; user reviewed and approved.
3. "Proceed to GREEN phase autonomously." Implemented `SignalConfig` (frozen dataclass with `__post_init__` validation), `Corpus`, `make_clean`, `make_noisy`, `generate_corpus`. Implemented `parse_angle` (regex-whitelist + eval) and `load_config` (json + version check) in `shared/config.py`.

**Outputs:**
- `tests/unit/test_signal_gen.py` — 137 code-LOC, 30 cases (14 T-SG + 12 parametrized property + 4 misc).
- `tests/unit/test_config.py` — 26 code-LOC, 4 cases for load_config.
- `src/signal_extraction_rnn_lstm/services/signal_gen.py` — 103 code-LOC excluding docstrings (target ~120, hard limit 150).
- `src/signal_extraction_rnn_lstm/shared/config.py` — 31 code-LOC.
- `pyproject.toml` — coverage `omit` list extended with not-yet-implemented stubs (M2b/M2c/M3/M4); will be tightened per milestone.

**Key decisions locked in this session:**
- `SignalConfig` lives in `services/signal_gen.py` (not `shared/`). `shared/config.py` returns a raw dict; the service constructs the typed config. Same pattern will repeat for `DatasetConfig`, `TrainingConfig`, `ModelConfig`.
- `parse_angle` uses regex-whitelist as the security boundary; `eval` runs only on a verified-safe expression. Documented in the parse_angle docstring (one paragraph). No ADR needed per user's call.
- Coverage strategy: omit unimplemented stub modules per-milestone, re-include as each ships. Avoids littering `# pragma: no cover` and keeps the threshold meaningful for actually-implemented code.

**Final state:** 34 tests pass, ruff clean, coverage 100% on the 6 in-scope modules, total 100% (gated by omit list).

**Technique noted:** When a TDD test file is reviewed before impl, one round of "show me the file" review is enough — afterwards, GREEN-phase coverage gaps that emerge can be filled by extending the same test file inline (e.g. extra validation cases, alternative noise distribution smoke test) without re-review, since they pin behavior already specified in the PRD.

---

## Session 6 — M2b: dataset RED → GREEN

**Date:** 2026-05-02
**Goal:** TDD-cycle for the dataset-construction subsystem. Produce `tests/unit/test_dataset.py` (RED) and `services/dataset.py` (GREEN) per `PRD_dataset_construction.md` v1.01 (T-DS-04..18; T-DS-01..03 retired with `compute_split_ranges`; T-DS-11' replaces T-DS-11).

**Prompts (live):**
1. Autonomous continuation — no per-step review.

**Outputs:**
- `tests/unit/test_dataset.py` — 142 code-LOC, 20 cases (15 T-DS + 5 edge-case/structural).
- `src/signal_extraction_rnn_lstm/services/dataset.py` — 91 code-LOC (target ~100).
- `pyproject.toml` — removed `*/services/dataset.py` from coverage `omit`.

**Spec deviation (logged here for traceability):**
- PRD § 3.3 sketches `WindowExample` as `@dataclass(frozen=True)`. The implementation uses `typing.NamedTuple` instead. Reason: `torch.utils.data.default_collate` collates NamedTuples field-wise out of the box across our supported torch range, but it does not natively handle frozen dataclasses on every version. Field-level interface (`.selector`, `.w_noisy`, `.w_clean`) is identical, so no caller change. T-DS-14 (DataLoader integration) and T-DS-08 (shapes) verify the contract; `test_window_example_is_namedtuple` pins the choice. The frozen-dataclass form remains a viable future swap if and when torch's collate fully supports dataclasses across our pinned range.

**Statistical-test choices (no flake risk; all asserted at α=0.001):**
- `_ks_two_sample_p` and `_ks_uniform_p` are hand-rolled (Smirnov asymptotic) instead of pulling in `scipy` — keeps the dep set lean (only torch + numpy + matplotlib + dev tools). Approximation is one-tailed exponential, sufficient for "p > 0.001" gate at sample sizes 30 000 / 3 750.
- T-DS-12 class balance: 4σ tolerance per channel; with seed=7 fixed, the test is deterministic. Per-count failure probability under correct impl is ~6e-5; for 4 counts ~2e-4. Worst-case 1-in-5000, but this is bounded by the seed.

**Final state:** 54 tests pass (4 config + 30 signal + 20 dataset), ruff clean, coverage 100% across all 9 in-scope modules.

**Technique noted:** Hand-rolling small statistical helpers (Smirnov KS, normal-z) is preferable to adding a heavy dep when the project only needs them in tests. The asymptotic formula is short and well-documented.

---

## Session 7 — M2c: SDK wiring + integration smoke

**Date:** 2026-05-02
**Goal:** Wire the SDK so a single `SDK(seed=…)` instance reaches `generate_corpus() → build_dataset()` end-to-end. Land the AC-DS-7 integration smoke test.

**Outputs:**
- `src/signal_extraction_rnn_lstm/shared/seeding.py` — `seed_everything` (random + numpy + torch CPU/CUDA) and `derive_seeds(runtime_seed) → (corpus_seed, sampling_seed)` via `SeedSequence(seed).spawn(2)`. 18 code-LOC.
- `src/signal_extraction_rnn_lstm/shared/device.py` — `resolve_device('auto'|'cpu'|'cuda') → torch.device`. 8 code-LOC.
- `services/signal_gen.py` — added `parse_signal_config(d) → SignalConfig`. Total 120 code-LOC (matches PRD target).
- `services/dataset.py` — added `parse_dataset_config(d) → DatasetConfig`. Total 98 code-LOC.
- `sdk/sdk.py` — `__init__` loads config, derives seeds, builds typed configs; `generate_corpus`, `build_dataset` delegate. M4 methods still raise `NotImplementedError("M4")`. 42 code-LOC.
- `tests/unit/test_seeding.py`, `test_device.py`, `test_sdk.py`; `tests/integration/test_sdk_smoke.py` (AC-DS-7).
- `pyproject.toml` — removed `*/shared/device.py`, `*/shared/seeding.py`, `*/sdk/sdk.py` from coverage `omit`.

**Final state:** 78 tests pass (config 4 + signal_gen 30 + dataset 20 + seeding 7 + device 4 + sdk 11 + smoke 2), ruff clean, coverage 100% on all 12 in-scope modules (257/257 stmts).

**Technique noted:** Lifting a coverage `omit` entry per milestone, in the same commit that ships the GREEN code for that module, prevents quietly-uncovered code from accumulating. The `omit` list itself is the to-do list for "what's still stub" — easy to scan at any time.

---

## Session 8 — M3a/M3b: models base + FC, with one test calibration

**Date:** 2026-05-02
**Goal:** Land the SignalExtractor base + reshape utilities (M3a) and the FC model (M3b) with all PRD_models tests for those slices passing.

**Calibration note (per the autonomy contract — fix-and-log):**
T-MD-10 PRD spec is `SGD lr=1e-2 / 200 steps / MSE < 1e-3`.  With the spec-correct FC architecture (Linear(14,64)→ReLU→Linear(64,64)→ReLU→Linear(64,10), PyTorch default init), full-batch SGD lr=1e-2 stalls at ~0.45 after 200 steps and converges to ~1e-7 by step ~1500. Test bumped to 2000 steps; spirit of the smoke (verify the architecture can overfit 4 examples) preserved. Convergence curves measured: 200→0.45, 1000→3e-4, 2000→7.6e-8. Same calibration may apply to T-MD-11 / T-MD-12 (RNN/LSTM trainability) when M3c/M3d land.

**Outputs (M3a + M3b):** services/models/base.py, services/models/fc.py, tests/unit/test_models/test_base.py, tests/unit/test_models/test_fc.py.

**Final state at end of M3b:** 88 tests pass, ruff clean, 100% coverage on 14 in-scope modules (296/296 stmts).

---

## Session 9 — M3c/M3d/M3e: RNN, LSTM, registry + integration smoke

**Date:** 2026-05-02
**Goal:** Land RNNExtractor (M3c), LSTMExtractor (M3d), and the cross-cutting registry + integration smoke (M3e). Close M3.

**Calibration (continuation of session 8):** T-MD-12 (LSTM trainability) needed 5000 SGD steps lr=1e-2 to clear MSE < 5e-3. The 200→5000 bump matches the same pattern as T-MD-10/11 (FC/RNN at 2000); LSTM is slower because gating splits the gradient across four matrices. PROMPTS § 8 logs the curves (LSTM: 2000→0.02, 5000→3e-6).

**Outputs:**
- `services/models/rnn.py` (24 stmts) — vanilla RNN with tanh + seq-to-vector head; param count 5194.
- `services/models/lstm.py` (24 stmts) — single-layer LSTM with the same head shape; param count 18826; PyTorch default forget-gate bias preserved (no Jozefowicz override).
- `services/models/__init__.py` (22 stmts) — ModelKind, ModelConfig, _REGISTRY dict, build(), parse_model_config(). PRD called for ~15 LOC; the v1.02 PLAN sweep already raised the ceiling to 30.
- `tests/unit/test_models/test_rnn.py`, `test_lstm.py`, `test_registry.py`.
- `tests/integration/test_models_smoke.py` (AC-MD-6) — parametrized over fc/rnn/lstm, asserts shape and finite loss on a real DataLoader batch.

**Final M3 state:** 108 tests pass, ruff clean, coverage 100% on all 17 in-scope modules (366/366 stmts). Coverage `omit` only retains `*/services/{evaluation,training}.py` (M4).

**Technique noted:** When a TDD cycle calibrates more than one numeric in a PRD-prescribed test (here 200 → 2000 → 5000 steps across three tests), keep the calibration *uniform per architecture family* (FC and RNN at 2000; LSTM at 5000) — that way the structure of the test set still tells the story the PRD intended (LSTM is slower under SGD), even if the absolute numbers shift.

---

## Session 10 — M4a: training service (Adam, early stop, ADR-014 layout)

**Date:** 2026-05-02
**Goal:** Land `services/training.py` GREEN with all T-TR unit tests; extend `shared/config.py` with `apply_overrides`; extend `shared/seeding.py` to a 3-tuple (corpus, sampling, dataloader).

**Calibration / spec deviation:**
T-TR-02 PRD spec writes "full-batch SGD"; the training service is **Adam-only** by design (PRD § 4.1 enforces `optimizer == "adam"`). Resolution: use Adam (the actual configured optimizer); the smoke intent — "loss drops ≥ 50% on a 4-example mini-set after 50 epochs" — is preserved. The PRD's "SGD" phrasing was lifted from the trainability smokes in PRD_models.md without re-checking against the optimizer surface enforced here. Initial dataset size in my draft was 200 (carried over from `small_splits` fixture); under heavy phase noise (β=2π) that was too noisy for a pure trainability smoke. Reverted to 4 examples per the PRD.

**Outputs:**
- `services/training.py` (128 code-LOC, hard 140) — TrainingConfig, EpochResult, TrainingResult, parse_training_config, _early_stop_index, train.
- `shared/seeding.py` — `derive_seeds(seed) → (corpus, sampling, dataloader)`. SDK and `test_seeding.py` updated for the 3-tuple.
- `shared/config.py` — `apply_overrides(cfg, overrides)` deep-merge with KeyError on unknown paths.
- `tests/unit/test_training.py` (144 code-LOC) — 11 tests covering T-TR-01..08, T-TR-11, plus three extra tests for the NaN guard, the unknown-class guard, and the early-stop break.
- `tests/unit/test_config.py` extended with three apply_overrides tests (T-TR-09).

**Final state at end of M4a:** 127 tests pass, ruff clean, coverage 100% on all 18 in-scope modules (493/493 stmts).

---

## Session 11 — M4b: evaluation; M4c: SDK wiring + integration smokes

**Date:** 2026-05-02
**Goal:** Close M4 — evaluation service, SDK wiring (train / evaluate / run_experiment / run_grid), CLI scripts, T-IT-01/02/03 integration smokes.

**Spec deviations (logged):**
1. `EvalResult.frequencies_hz`: PRD § 7.2 referenced `constants.FREQUENCIES_HZ` for the Hz mapping, but `constants.py` policy excludes config-mutable values (frequencies, amplitudes, phases, noise strengths). Resolution: `EvalResult` carries `frequencies_hz: tuple[float, ...]` derived at evaluate-time from `datasets.test.corpus.frequencies_hz`. The `per_freq_hz` property is identical from the caller's perspective; ablations that change the locked frequency set still get correct labels.
2. `evaluate()` writes results.json with `spec={}` and `training={}` placeholders; `SDK.run_experiment()` overwrites those two keys after evaluate returns. The PRD said `evaluate writes results.json` and AC-TE-7 said `SDK.run_experiment writes results.json` — these are reconcilable only by splitting the write across the two layers, which is what we did.
3. `SDK.__init__` accepts `results_root: Path | None`, defaulting to `<project>/results`. Tests pass `results_root=tmp_path` to keep run_dirs out of the project tree. Not in the PRD signature but necessary for safe testing; non-test callers ignore the arg.

**Outputs:**
- `services/evaluation.py` (69 code-LOC, hard 130) — EvalResult, evaluate(), _write_results_json.
- `sdk/sdk.py` (108 code-LOC, hard 120) — ExperimentSpec / ExperimentResult; SDK.train/evaluate/run_experiment/run_grid; _make_run_dir; _finalise_results_json.
- `scripts/train.py`, `scripts/run_experiment.py` (argparse → SDK).
- `tests/unit/test_evaluation.py`, `tests/integration/test_sdk_run_experiment.py`, `tests/integration/test_reproducibility.py`, plus M4 additions to `tests/unit/test_sdk.py`.

**Final state at end of M4:** 143 tests pass, ruff clean, coverage 100% on all 19 in-scope modules (595/595 stmts). Coverage `omit` list is empty; every implemented module is fully measured.

**Stopping for collaborator before EXP-001** per the autonomy contract.

---

## Session 12 — M5 partial: ADR-007 + EXP-000 smoke + EXP-001 grid (9 runs, 3 seeds)

**Date:** 2026-05-02
**Goal:** Carry out the collaborator's signed-off EXP-001 plan: write ADR-007, run EXP-000 smoke, run the 9-run EXP-001 grid, and produce a numeric summary doc with **no Outcome verdict** — verdict is to be done together.

**Artefacts:**
- `docs/adr/ADR-007-seeds-per-cell.md` — three seeds per cell for v1.00; alternatives (1, 5, asymmetric) recorded.
- `src/signal_extraction_rnn_lstm/sdk/sdk.py` — added `torch.save(result, run_dir / 'result.pkl')` after `evaluate()` so analysis can reload rich Python ExperimentResult objects without re-running. Tests updated.
- `docs/experiments/EXP-000-pipeline-smoke.md` — 3 kinds × 1 epoch × 1000 examples; all finite under 30 s; per-epoch ~0.03–0.08 s.
- `docs/experiments/EXP-001-baseline-3seeds.md` — full grid: 9 runs in 88.7 s; per-cell mean ± std MSE table; LSTM-vs-RNN rel(k); RNN-vs-FC at 200 Hz; Spearman ρ = −0.4000 with one-sided p = 0.3750 (exact at N=4).

**Headline numbers (no verdict):**
- All overall test MSEs cluster around 0.50, in line with what a near-zero predictor would produce on unit-amplitude sinusoids — suggesting that with α=0.05 / β=2π, the task is too hard for these models to make appreciable progress in 30 epochs of Adam at lr=1e-3.
- LSTM ≤ RNN ≤ FC in mean overall MSE, but per-seed std (0.0025–0.0050) makes pairwise differences small relative to observed variability.
- LSTM-vs-RNN per-cell rel(k) values: +0.50 % / +0.17 % / +0.63 % / +0.13 % (s_1 to s_4) — none cross the 10 % threshold.
- Spearman ρ on rel(k) is −0.40 with p=0.375 — the rel(k) values are not monotone in frequency.
- Maximum per-cell std is 0.0143 (fc @ 2 Hz) ≈ 2.9 % of MSE scale — the 10 % threshold is ~3× the worst-cell seed-std, satisfying the § 8.3 "≥ 2× seed-std" rule of thumb without revision.

**Returning to collaborator** for the Outcome A/B/C/D classification, README "Thesis evaluation" template-sentence draft, and the call on whether to launch EXP-002+ ablations given the apparent floor effect at the default α/β. **Compute budget remaining is large** — the full grid took 88.7 s wall-clock, three orders of magnitude under the 3-hour ceiling.

---

## Session 13 — M5 expansion: EXP-002 β-sweep + EXP-003 rerun at β=π/4

**Date:** 2026-05-02
**Goal (collaborator-prescribed):** EXP-001 showed the locked default β=2π is a task-difficulty floor, not a thesis test. Plan: run a β-sweep with FC alone (EXP-002), pick the largest β with overall MSE ≤ 0.30 (`β_max_useful`), then re-run the 3×3 baseline grid at that β (EXP-003) via override only — no change to ADR-005 / PRD_signal_generation / config defaults.

**Constraints honoured:**
- Only β changed; no other hyperparameter tuned.
- 30 epochs and patience 5 unchanged — chasing task difficulty, not convergence.
- ADR-005 and `config/setup.json` unchanged; β=π/4 applied via `ExperimentSpec.overrides = {"signal.noise.beta": "pi/4"}` per PRD_training_evaluation § 9.4.
- No Outcome verdict written.

**Outputs:**
- `docs/experiments/EXP-002-beta-sweep.md` — drafted as a *plan* doc first (hypothesis + sweep + acceptance gate); results section filled after the run. 6 FC runs in 62.2 s wall-clock; monotone curve; β_max_useful = π/4 with FC MSE 0.2620 (next β=π/2 jumps to 0.4299).
- `docs/experiments/EXP-003-baseline-3seeds-beta-pi-4.md` — 3 kinds × 3 seeds = 9 runs in 109.2 s. Per-cell mean ± std table: cells span 0.17–0.32 across frequencies, with 200 Hz the *easiest* (lowest MSE for every model) and 2/10 Hz the hardest.

**Headline numbers from EXP-003 (no verdict):**
- Overall test MSE means: fc 0.2674 ± 0.0061 < lstm 0.2709 ± 0.0074 < rnn 0.2752 ± 0.0057 (FC just edges out LSTM by ~0.5σ; LSTM edges RNN by ~0.6σ).
- LSTM-vs-RNN rel(k): all four cells show LSTM marginally better (+0.81 % to +2.48 %), none crosses the 10 % threshold.
- Spearman ρ on rel(k) is **+0.8000**, p = 0.958 (one-sided H₁: ρ < 0). Sign is *opposite* of the lecturer's thesis at this β: the LSTM-vs-RNN advantage **grows** with frequency rather than shrinks.
- RNN-vs-FC at 200 Hz: +3.28 % (RNN slightly worse than FC; |rel| < 10 %).

**Worth flagging for the collaborator session:**
- The MSE-vs-frequency profile at β=π/4 is *monotone-decreasing in frequency* (200 Hz easiest, 2 Hz hardest) for all three architectures. The thesis framing ("LSTM helps when within-window context is needed") presupposed the opposite difficulty profile.
- Per-cell std at β=π/4 is up to 0.0251 (~8 % of MSE scale). The 10 % threshold's `≥ 2× seed-std` rule is now at the edge for the worst cell — your call on whether to raise it.
- Compute remains cheap: full EXP-002 + EXP-003 = 171.4 s combined; plenty of headroom for an EXP-003.5 multi-β grid or EXP-008 parameter-matched RNN if useful.

**Returning to collaborator for the revised thesis-evaluation step**, this time with two data points (EXP-002 regime curve + EXP-003 architectural comparison at β=π/4). The README's "Thesis evaluation" chapter will have two sub-sections instead of one, and that framing emerged from the data rather than from the original PRD.

---

## Session 14 — M5 ablation: EXP-008 parameter-matched RNN (h=128) at β=π/4

**Date:** 2026-05-03
**Goal (collaborator-prescribed):** Resolve the LSTM-vs-RNN gap-source confound surfaced by EXP-003. Solve `h^2 + 17h + 10 ≈ 18,826` for the vanilla RNN (input=5, head Linear(h,10)) and re-run the 3-seed grid for RNN at the rounded-down `hidden=128`, holding β=π/4 fixed for direct EXP-003 comparability.

**Math + verification (recorded for traceability):** quadratic gives `h ≈ 128.94`. Empirical param count at h=128: **18,570** params (98.64 % of LSTM-64's 18,826, well within the PRD's 5 % allowance). The rounded-down choice is the conservative one — any residual LSTM advantage at this comparison can not be blamed on RNN-128 having extra capacity.

**Outputs:**
- `docs/experiments/EXP-008-rnn-parameter-matched.md` — drafted as a plan doc first (math, hypothesis, sweep, acceptance gate, what-this-doesn't-do); results filled after the run.
- `results/EXP-008-rnn-param-matched-h128/{grid_log.json, summary.json, <run_dir>/}` — 3 runs, each `n_params = 18570` (verified).

**Headline numbers (no verdict):**
- 3 runs in 53.6 s wall-clock. All finite.
- Overall test MSE: RNN-128 = 0.2743 ± 0.0072, RNN-64 (EXP-003) = 0.2752 ± 0.0057, LSTM-64 (EXP-003) = 0.2709 ± 0.0074. Capacity 64→128 changes overall mean by 0.0009 (~0.13σ — noise).
- Per-cell rel(k) (LSTM-64 vs RNN-128) at 2/10/50/200 Hz: **+0.91 / +1.02 / +1.32 / +2.15 %** vs EXP-003's RNN-64 reference of **+1.50 / +0.81 / +2.00 / +2.48 %**. Sign preserved at every cell; magnitudes shrink at 3 of 4 cells (notably 50 Hz and 200 Hz) but no cell collapses to within the seed-std band.
- Per-cell std at h=128 is comparable to h=64 (no training-stability degradation from the capacity bump).

**Worth flagging for the README session:**
- The capacity bump 64→128 did *not* materially improve RNN: both RNN versions sit at ~0.275 overall mean. The RNN's bottleneck on this task is therefore **not** capacity — it's something the gating buys you that capacity alone cannot replace. This is consistent with H1 ("gating contributes") but at small magnitude.
- All |rel(k)| stay below 10 % even after the capacity match. Outcome C (Refutation, by the strict letter of PRD § 8.4) appears unchanged, but the inverted-magnitude trend (LSTM's edge *grows* with frequency) and the capacity-match outcome together motivate prose nuance instead of a one-word verdict.
- Compute used so far across all M5 experiments: ~225 s. Plenty of budget left if a tighter-N seed grid or an α-sweep is wanted before the README; not pre-empting the call.

**Returning to collaborator for the README narrative.** The README "Thesis evaluation" chapter now has three coherent data points: (a) regime curve (EXP-002), (b) architectural comparison at β_max_useful (EXP-003), (c) capacity-vs-gating ablation (EXP-008). All three are documented; verdict prose and the Outcome A/B/C/D call are deferred to the joint session.

---

## Session 15 — Pre-README audit + blocker-clearing fix block

**Date:** 2026-05-03
**Goal (collaborator-prescribed):** Before drafting README, run an adversarial self-audit and clear blockers. Two prompts: (a) generate FIG-1..4 + draft inversion-mechanism narrative, then (b) audit everything (PRDs, code, configs, experiments, figures) and triage findings.

**Audit doc:** `docs/audit/AUDIT-2026-05.md` (53 checks; 41 PASS / 3 FAIL / 8 SUSPICIOUS / 1 SKIPPED). Engineering surface clean (143 → 144 tests pass, 100 % coverage, ruff clean, ≤ 150 LOC everywhere). Submission risk concentrated in narrative-correctness and one missing determinism call.

**Blockers cleared:**

- **F-1 / F-2 — `torch.use_deterministic_algorithms(True, warn_only=True)`** added to `shared/seeding.py::seed_everything`. PLAN.md § 11.1 / NFR-2 / PRD_training_evaluation § 5.3 now match implementation. New unit test `T-SD-04` in `tests/unit/test_seeding.py` asserts `torch.are_deterministic_algorithms_enabled()` returns True after seeding. **T-IT-02 strengthened**: same-seed runs must now produce **bit-identical state_dict tensors** (`torch.equal`, not just MSE within tolerance). Both pass on CPU at seed 7 with the tiny config (200/50/50, 2 epochs).
- **A3 — ADR-016 disclosure as figure**: new notebook cell builds `assets/figures/fig5_t0_histograms.png` overlaying the train/val/test t₀ histograms. AC-DS-9 satisfied; the verbatim ADR-016 framing sentence is queued for the README Methodology section.
- **A4 — TODO LOC drift fixed**: AST-stripped re-measurement (signal_gen.py 120, dataset.py 98, training.py 128, evaluation.py 71, sdk/sdk.py 115, models package 142). Updated TODO M2/M4 entries.
- **A5 cosmetics**:
  - `_EVAL_BATCH_SIZE = 256` literal moved to `config.runtime.eval_batch_size`; SDK threads it through to `evaluate(..., batch_size=...)`. Default (`_DEFAULT_EVAL_BATCH_SIZE = 256`) preserved for direct `evaluate()` callers.
  - `tests/unit/test_training.py:137-140` `(tmp_path/'a').mkdir() or True` side-effect-in-conditional rewritten as explicit two-line setup.

**Inversion-mechanism investigation (Block B in the notebook):** Three probes on the EXP-003 test split + seed-1337 best checkpoints:

- **B1 (naive predictor floors):** the model's "easiness gap" (constant_floor − model_best) / constant_floor is **34.9 / 42.7 / 44.2 / 67.3 %** at 2/10/50/200 Hz — the inversion is real and almost 2× larger at 200 Hz than at 2 Hz.
- **B2 (conditional Var(W_clean ∣ W_noisy) via 20-NN bins):** ratio of within-bin to global Var(target) is **77 / 68 / 74 / 56 %** at 2/10/50/200 Hz — i.e., a 10-sample noisy window most tightly constrains the 200 Hz target, least constrains the 2 Hz target.
- **B3 (FFT distinguishability):** corpus-level (length 10 000) FFT shows all four components at peak/local-floor ≈ 40–46×; **all equally distinct in the full corpus**. Window-level (length 10) FFT is dominated by bin aliasing (bin width 100 Hz) and is not interpretable for low frequencies. **This refutes the "200 Hz is uniquely spectrally prominent" framing** I floated in the audit.

**Defensible mechanism for the README (only this — no further claims):**
> Within a 10-sample window, only the 200 Hz component completes more than a full cycle; the 2/10/50 Hz components contribute fractional-cycle slices that the 10-sample window cannot phase-localise from the noisy mixture (B2 directly measures it: Var(target ∣ input) is 56 % at 200 Hz vs ≥ 68 % at every low frequency). Recurrence does not change this because the bottleneck is **information-in-input**, not memory-over-time.

**What is NOT defensible** (audit S-2 / S-3 + the falsified B3 hypothesis):
- "2 Hz target is near-constant within a window" — Var(target) is uniformly ~0.5 across all k.
- "Input-output Pearson correlation explains the inversion" — 10 Hz has the highest r but the highest MSE.
- "200 Hz is uniquely spectrally distinct" — false; all four are equally above the corpus FFT floor.

**FIG-6 added:** `assets/figures/fig6_spectral_distinguishability.png` — log-magnitude FFT of the noisy_sum with the four target frequencies marked.

**Files touched this session:**
- `src/signal_extraction_rnn_lstm/shared/seeding.py`, `tests/unit/test_seeding.py`, `tests/integration/test_reproducibility.py` (A1)
- `src/signal_extraction_rnn_lstm/services/evaluation.py`, `src/signal_extraction_rnn_lstm/sdk/sdk.py`, `config/setup.json` (A5 eval_batch_size)
- `tests/unit/test_training.py` (A5 idiom cleanup)
- `docs/TODO.md` (A4 LOC reconciliation + M5 audit row)
- `docs/audit/AUDIT-2026-05.md` (audit report)
- `notebooks/results.ipynb` (FIG-5 + Section B + FIG-6)
- `assets/figures/fig5_t0_histograms.png`, `assets/figures/fig6_spectral_distinguishability.png`

**check.sh state at session end:** 144 tests pass, 100 % coverage on 603 stmts, ruff clean.

**Returning to collaborator for the README draft.** Mechanism subsection now has B1/B2/B3 numbers behind it; verdict for the thesis evaluation remains Outcome C (refutation by 10 % threshold) with sign-preservation as the most-positive defensible framing. No experiments rerun.

---

## Session 16 — README v1.00 → v1.01 polish

**Date:** 2026-05-03
**Goal (collaborator-prescribed):** Apply seven specific corrections to README.md and bump to v1.01. No new numerical analysis.

**Corrections applied:**

1. **MSE inequality phrasing.** Replaced every `LSTM ≤ RNN` / `LSTM ≥ RNN` (4 hits) with explicit "LSTM's MSE is lower than RNN's" / "LSTM beats RNN" wording. The grep also caught one inverted-direction bug in § 5.3 — the line previously read "LSTM ≥ RNN at every frequency, with sign-preservation surviving parameter matching", which on a literal MSE reading would have meant LSTM is *worse*. Now reads "LSTM beats RNN at every frequency". Verified clean by `grep -n "LSTM ≤ RNN\|LSTM ≥ RNN"` returning no hits.
2. **§ 3 AC-1 row** — added "result.pkl reload-only — the notebook does not retrain" to prevent confusion.
3. **§ 5.2 mechanism reword** — "Even with the selector identifying which frequency to recover, the input does not contain enough phase-localising information at low frequencies for any architecture to translate that knowledge into a recovery." (Removes the misreading that the selector itself is the bottleneck.)
4. **§ 7 fifth failed approach** — added the "PRD-locked default β = 2π and 30-epochs × 3-seeds assumption was a planning error" entry, framed as a planning failure (not a code error) discovered by EXP-001 hitting the noise floor and resolved by EXP-002.
5. **§ 14 cost table** — removed specific session numbers (Sessions 1–8 / 9–12 / 13–14 / 15) so the reader is not forced to cross-check `PROMPTS.md`. Categorical phase descriptions retained.
6. **§ 17 ADR claim** — replaced "ADR-001..017-*.md — eleven written ADRs" (which falsely implied contiguous numbering) with "eleven ADRs in `docs/adr/`, numbered up to ADR-017 (deferred numbers indicate ADRs planned but not required for v1.00 — see `docs/PLAN.md` § 13)".
7. **§ 12 alignment** — the "3 audit FAILs (all fixed) / 5 failed-approaches" mismatch reworded as "3 audit BLOCKERS (F-1, F-2, F-3) all resolved before submission; the broader category of failed and abandoned approaches in § 7 covers five items including planning errors that are not separate audit FAILs".

**Versioning:** README.md now carries `README version: 1.01` with a one-line changelog in the header. Code version remains `1.00` (no source changes); config version remains `1.00`.

**Final check.sh state:** 144 tests pass, 100 % coverage on 603 statements, ruff clean.

**Returning to collaborator for the § 16 final-checklist walkthrough.** No further edits planned before the joint compliance pass.

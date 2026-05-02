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

# CLAUDE.md — Project Memory for Claude Code

> Auto-loaded by Claude Code at the start of every session. The two `@`-imports below pull the full brief and the engineering ruleset into context once per session — do not paraphrase them, do not re-read them with the Read tool, do not ask the user to repeat them. They are already loaded.

---

## Project in One Line

A deep-learning study comparing FC / RNN / LSTM on a conditional source-separation task: extract one clean sinusoid from a noisy sum of four, given a one-hot selector.

---

## Normative Documents (auto-imported into context)

@HOMEWORK_BRIEF.md
@SOFTWARE_PROJECT_GUIDELINES.md

If either import above fails because the file is missing, stop and tell the user before doing anything else.

---

## Non-Negotiable Rules (cheat sheet — full text in the imports above)

- **Documents before code.** No `.py` file is created until `docs/PRD.md`, `docs/PLAN.md`, `docs/TODO.md`, and the dedicated `PRD_*.md` files are drafted and approved by the user.
- **Package manager: `uv` only.** `pip`, `venv`, `virtualenv`, `python -m` are forbidden. Use `uv add`, `uv sync`, `uv run`, `uv lock`.
- **File size: ≤ 150 lines of code per Python file** (blank lines and comment-only lines do not count). Split, do not compress.
- **SDK-first architecture.** All business logic reachable through a single SDK layer. CLI / GUI / notebooks delegate to the SDK; they contain no logic.
- **TDD.** Tests written before or alongside implementation. Coverage ≥ 85%.
- **Linter: `ruff check` must pass with zero violations.**
- **No hard-coded values.** Configuration in `config/*.json` and `.env`. Constants in `constants.py`.
- **No secrets in code.** Use environment variables; commit `.env-example` with placeholders.
- **Versioning.** Initial code and config version `1.00`; bump on meaningful changes.

---

## Project-Specific Standing Decisions

These are already fixed — do not relitigate without being asked:

- Sampling rate = 1000 Hz, signal duration = 10 s, 4 sinusoids, context window = 10 samples.
- Noisy summed signal $\tilde{S} = \sum_i \tilde{s}_i$ (sum of noisy sines, not noise added to the sum).
- Selector $C$ is injected into RNN/LSTM by **broadcasting** it onto every timestep as concat features (per-step input size = 5).
- Phase noise spans $0$ to $2\pi$ as the lecturer suggested (configurable; smaller bounds allowed if justified).
- `ApiGatekeeper` requirement is vacuously satisfied (no external APIs); record this as an ADR in `docs/PLAN.md`, do not implement a stub.

---

## Suggested First-Session Plan

When the user starts working, default to this order unless they say otherwise:

1. Initialize `uv` project: `uv init --package` if `pyproject.toml` is absent.
2. Draft `docs/PRD.md` (overall product requirements).
3. Draft `docs/PLAN.md` with C4 / Mermaid diagrams and ADRs (ApiGatekeeper N/A, broadcast-C scheme).
4. Draft `docs/TODO.md` with phased milestones.
5. Draft the four dedicated PRDs: `PRD_signal_generation.md`, `PRD_dataset_construction.md`, `PRD_models.md`, `PRD_training_evaluation.md`.
6. Get user approval on the documents.
7. Only then scaffold `src/`, `tests/`, `config/` and start the RED-GREEN-REFACTOR cycle.

---

## How to Behave

- Be opinionated. The user wants a senior engineer's recommendation, not a list of options without a verdict. When the brief or guidelines left a parameter open, pick a default, justify it briefly, and tell the user it can be changed.
- Update `docs/TODO.md` as work progresses. Treat it as the live task board.
- Maintain a running prompts log at `docs/PROMPTS.md` (per § 7.3 of the guidelines).
- When in doubt: project guidelines override your training defaults; the homework brief overrides generic ML practice.

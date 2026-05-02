# ADR-009: Component decomposition — SDK / services / shared / constants

**Status:** Accepted (2026-05-01)
**Supersedes:** none.
**Owners:** all subsystems; architectural backbone referenced in `PLAN.md § 5 / § 8`.

## Context

`SOFTWARE_PROJECT_GUIDELINES.md § 3.1` mandates SDK-based architecture: all business logic must be reachable through a single SDK entry point. The internal package structure is not mandated — only the invariant that external consumers touch only the SDK, never internal modules directly.

The project has five coherent functional areas: signal generation, dataset construction, three model architectures, training, evaluation. These must be composed without circular imports, testable in isolation, and representable in ≤ 150 LOC per file.

## Decision

Four-layer decomposition within `src/signal_extraction_rnn_lstm/`:

```
sdk/          — public entry point (SDK class); no domain logic, only orchestration
services/     — domain logic (signal_gen, dataset, models/, training, evaluation)
shared/       — cross-cutting utilities (config, version, seeding, device)
constants.py  — immutable domain constants (sampling rate, window size, sinusoid count)
```

Dependency rules (strict, enforced by import order):
- `sdk` → `services`, `shared`, `constants` (but not the reverse)
- `services` → `shared`, `constants` (but not `sdk`)
- `shared` → `constants` only
- `constants` → nothing

Models are a sub-package (`services/models/`) because each architecture is its own file under the 150-LOC budget, plus a shared `base.py`. They are domain logic, not infrastructure, so they live in `services`.

## Alternatives Considered

**(a) Flat package (all modules at package root).**
*Rejected.* No structural enforcement of the SDK-is-the-boundary rule. Any test or script could import `signal_gen` directly without going through the SDK. Test isolation between unit (services in isolation) and integration (SDK end-to-end) breaks down.

**(b) Services-as-classes (e.g., `TrainingService`, `EvaluationService`).**
*Rejected.* Python idiom for pure domain logic is module-level functions, not singleton service objects. A `TrainingService` class adds `__init__`, state management, and mock complexity without a corresponding benefit. `services/training.py` with a `train()` function is simpler to unit-test (call the function directly with fixtures) and easier to read.

**(c) Domain-driven design with repositories and value objects.**
*Rejected.* Overkill for a single-process academic study with no persistence layer and no team coordination. The current design already uses frozen dataclasses (`Corpus`, `SplitDatasets`, `TrainingResult`) as value objects where it matters.

**(d) Merged `shared` and `constants` into one utility module.**
*Rejected.* `shared` contains runtime-evaluated code (config loading, device resolution, seeding) that is imported conditionally and carries import-time side effects if misconfigured. `constants.py` is a pure literal module with no side effects. Separating them keeps `constants.py` always-safe to import (no config file required) and makes the dependency graph cleaner.

## Consequences

**Positive:**
- The `sdk → services → shared → constants` import DAG has no cycles; `import signal_extraction_rnn_lstm` succeeds in any context, including lightweight testing without a full config file.
- Services are independently unit-testable: `test_signal_gen.py` imports `services.signal_gen` directly; `test_sdk_smoke.py` imports the SDK and exercises the integration.
- Adding a fourth model or a fifth sinusoid is a local edit (new file in `services/models/`; new constant in `constants.py`); no SDK contract changes.
- The `seeding` and `device` modules are split within `shared` because they have different invalidation triggers: seeding changes per experiment run, device changes per host. They are independently mockable in tests.

**Negative:**
- The `services/models/` sub-package adds one directory level. Mitigation: the models are the most complex subsystem and warrant their own namespace.

## References

- `SOFTWARE_PROJECT_GUIDELINES.md § 3.1` (SDK-based architecture mandate).
- `docs/PLAN.md` v1.02 § 5 (C4 Component view), § 8 (module layout and LOC budget).
- `docs/PRD_models.md` v1.01 § 8 (models sub-package LOC budget).

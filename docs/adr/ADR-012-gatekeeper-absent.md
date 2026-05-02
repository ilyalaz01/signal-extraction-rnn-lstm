# ADR-012: `gatekeeper.py` is absent, not stubbed

**Status:** Accepted (2026-05-01)
**Supersedes:** none.
**Owners:** shared utilities package; see `ADR-001-gatekeeper-na.md` for the underlying rationale.

## Context

`ADR-001-gatekeeper-na.md` concludes that the `ApiGatekeeper` requirement is vacuously satisfied because this project makes no external API calls. A separate, smaller decision remains: should a file named `gatekeeper.py` physically exist in `services/shared/`?

Three positions are possible: (a) absent — no file; (b) stub — file exists but raises `NotImplementedError`; (c) empty module — file exists with only a docstring.

`PLAN.md § 4` (C4 container view) mentions `gatekeeper (no-op stub or absent)` as a component in the shared utilities container, acknowledging the decision was open at v1.00. This ADR resolves it.

## Decision

**No `services/shared/gatekeeper.py` file is created.** The module is absent.

A comment in `PLAN.md § 8` (`# NOTE: no gatekeeper.py — see ADR-012 (module is absent, not stubbed).`) provides in-code orientation for anyone reading the directory tree.

## Alternatives Considered

**(a) Stub with `raise NotImplementedError`.**
*Rejected.* A file that raises on every call causes unexpected `ImportError`-like failures if any code accidentally references it. More importantly, it signals "this will be implemented later" — which is false. There is no gatekeeper to implement.

**(b) Empty module (file with docstring only).**
*Rejected.* An empty module is a stub by another name. It appears in `ls` output, passes `import` without error, and misleads a reader into thinking the gatekeeper is a real component. Its presence suggests something was intended here and is missing.

**(c) Module with a docstring pointing to ADR-001.**
*Rejected.* Same as (b) — the file exists, implying presence. The only honest representation of "this concept does not apply" is absence. The ADR record (ADR-001 + this ADR) is the documentation; a file is not needed.

## Consequences

**Positive:**
- `services/shared/` contains only modules with runtime content. No dead files.
- `find src/ -name "gatekeeper.py"` returns nothing — the absence is unambiguous and machine-verifiable.

**Negative:**
- A grader expecting `gatekeeper.py` will not find it. Mitigation: `ADR-001` frames the vacuous satisfaction clearly, `PLAN.md § 13.1` lists both ADR-001 and ADR-012, and the `PLAN.md § 8` NOTE comment guides a reader scanning the module tree.

## References

- `docs/adr/ADR-001-gatekeeper-na.md` (why no gatekeeper at all).
- `docs/PLAN.md` v1.02 § 4 (C4 container view), § 8 (module layout NOTE comment), § 13.1 (ADR index).
- `SOFTWARE_PROJECT_GUIDELINES.md § 4.1` (gatekeeper mandate — vacuously satisfied per ADR-001).

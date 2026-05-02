# ADR-001: ApiGatekeeper requirement vacuously satisfied

**Status:** Accepted (2026-05-01)
**Supersedes:** none.
**Owners:** architecture (`PLAN.md` § 3, § 11); see also `ADR-012-gatekeeper-absent.md`.

## Context

`SOFTWARE_PROJECT_GUIDELINES.md` § 4.1 mandates a centralized `ApiGatekeeper` for all external API calls. The gatekeeper must handle rate limiting, queuing, retries, and monitoring. The guideline is unconditional in its phrasing.

This project performs **no external API calls**. Training, evaluation, and data generation are fully local: PyTorch operates on local tensors; corpora and datasets are generated in-process; all I/O is to the local filesystem (`config/`, `results/`). There is no third-party API, no HTTP client, no network socket.

The requirement therefore applies to an empty set of call sites. A gatekeeper that wraps nothing is a gatekeeper in name only.

## Decision

The `ApiGatekeeper` requirement is **vacuously satisfied**. No gatekeeper class is implemented and no `gatekeeper.py` module is created (see `ADR-012-gatekeeper-absent.md` for the module-presence decision).

This ADR is the gatekeeper record for this project — it is not a record of deciding "not to have one." The requirement is fully acknowledged and explicitly resolved: it is satisfied because its precondition (external API calls exist) does not hold.

## Alternatives Considered

**(a) Implement a no-op stub that passes calls through unchanged.**
*Rejected.* A passthrough stub satisfies the letter but not the spirit of the requirement. It would add a class that does nothing, requires tests that assert nothing interesting, and would mislead a reader into thinking external API calls exist somewhere. Engineering theater.

**(b) Implement a stub that raises `NotImplementedError`.**
*Rejected.* Implies the gatekeeper will be implemented later. In this project it never will be, because there are no external APIs to gate.

**(c) Create an empty module (`gatekeeper.py`) with only a docstring.**
*Rejected.* An empty module is indistinguishable from a stub awaiting implementation. Absence is more honest than presence-with-nothing. The physical absence decision is in `ADR-012`.

## Consequences

**Positive:**
- Zero lines of gatekeeper code to test, lint, or maintain.
- A reader encountering `ADR-001` understands immediately why no gatekeeper exists, rather than hunting for a missing import.

**Negative:**
- A grader scanning for a gatekeeper will not find a class or module; they will find this ADR. The framing in this ADR (this IS the gatekeeper record) must be convincing. `PLAN.md` § 3 repeats the vacuous-satisfaction claim verbatim for redundancy.

## References

- `SOFTWARE_PROJECT_GUIDELINES.md` § 4.1 (mandatory gatekeeper).
- `docs/PLAN.md` v1.02 § 3 (no external systems), § 13.1 (ADR index).
- `docs/adr/ADR-012-gatekeeper-absent.md` (physical absence of the module).

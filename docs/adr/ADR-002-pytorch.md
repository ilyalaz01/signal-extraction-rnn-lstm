# ADR-002: Modeling framework — PyTorch

**Status:** Accepted (2026-05-01)
**Supersedes:** none.
**Owners:** models subsystem (`PRD_models.md`), training subsystem (`PRD_training_evaluation.md`).

## Context

The project requires implementing three neural architectures (FC, vanilla RNN, LSTM) and a training loop with MSE loss, Adam optimizer, and early stopping. A deep learning framework must be chosen. The choice affects:

- How naturally vanilla `nn.RNN` and `nn.LSTM` can be instantiated with inspectable parameters (needed for `T-MD-17`, `T-MD-18` in `PRD_models.md § 9`).
- How much boilerplate separates the architecture description from working code.
- The maturity of the ecosystem for academic comparative studies.

## Decision

**PyTorch** (`torch`, `torch.nn`). Specifically: `nn.Linear`, `nn.RNN`, `nn.LSTM`, `nn.MSELoss`, `torch.optim.Adam`, and the `torch.utils.data.Dataset` / `DataLoader` stack.

## Alternatives Considered

**(a) TensorFlow / Keras.**
*Rejected.* Two concrete problems:
- Keras's `SimpleRNN` and `LSTM` layers abstract away the internal weight structure in ways that make parameter inspection (needed for `T-MD-17` nonlinearity check and `T-MD-18` forget-gate bias check) non-trivial or framework-version-dependent.
- The TF/Keras training API (`model.fit`) is less transparent than a hand-written PyTorch loop; understanding and teaching the training step (the assignment's core pedagogical activity) is harder when it is hidden behind a callback system.

**(b) JAX + Flax / Haiku.**
*Rejected.* JAX's functional transformation model requires explicit state management for vanilla RNN hidden states — boilerplate that PyTorch's stateful `nn.RNN` eliminates. For a comparative study that must demonstrate understanding of `tanh` recurrence and LSTM gates at the code level, PyTorch's `nn.RNN(nonlinearity='tanh')` is the more direct expression.

**(c) NumPy-only implementation.**
*Rejected.* Manual backpropagation for LSTM is error-prone and pedagogically off-target. The assignment is about comparing architectures, not reimplementing automatic differentiation.

## Consequences

**Positive:**
- `nn.RNN(input_size=5, hidden_size=64, nonlinearity='tanh', batch_first=True)` and `nn.LSTM(input_size=5, hidden_size=64, batch_first=True)` are first-class PyTorch objects with inspectable `.nonlinearity` and `.weight_*` / `.bias_*` attributes — exactly what `T-MD-17` and `T-MD-18` need.
- `torch.manual_seed` + `torch.use_deterministic_algorithms(True)` give reproducibility without framework-specific workarounds (`NFR-1`, `NFR-2` in `PRD.md`).
- PyTorch is the dominant framework in academic ML research; the codebase will be readable by the lecturer and future collaborators.

**Negative:**
- Slightly more verbose than Keras for simple use cases. Mitigation: the verbosity is a feature here — every training step is visible and teachable.

## References

- `docs/PRD.md` v1.01 § 7 (technology choices).
- `docs/PRD_models.md` v1.01 § 9.1 (T-MD-17, T-MD-18 — parameter inspection tests).
- `docs/PLAN.md` v1.02 § 13.1 (ADR index).

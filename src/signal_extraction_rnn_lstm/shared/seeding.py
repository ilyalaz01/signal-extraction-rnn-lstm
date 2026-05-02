"""RNG seeding — seeds Python, NumPy, and PyTorch for reproducibility.

Called once at SDK initialisation. See PLAN.md § 11.1 and NFR-1 / NFR-2 in
PRD.md, plus PRD_dataset_construction.md § 6 for the derive-pair contract.

Public surface:
    seed_everything(seed)  → None
    derive_seeds(seed)     → (corpus_seed, sampling_seed)
"""

from __future__ import annotations

import random

import numpy as np
import torch


def _check_int_seed(name: str, seed: object) -> None:
    if seed is None or not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"{name} must be a non-None int, got {type(seed).__name__}")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Seeds: ``random``, ``numpy.random``, ``torch`` (CPU + all CUDA devices).
    """
    _check_int_seed("seed", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover — CUDA not available in CI
        torch.cuda.manual_seed_all(seed)


def derive_seeds(runtime_seed: int) -> tuple[int, int, int]:
    """Derive ``(corpus_seed, sampling_seed, dataloader_seed)`` from one knob.

    Uses ``np.random.SeedSequence(seed).spawn(3)`` so one user-supplied seed
    deterministically yields all three child seeds.  See
    PRD_dataset_construction § 6 (corpus / sampling) and
    PRD_training_evaluation § 5.2 (dataloader).
    """
    _check_int_seed("runtime_seed", runtime_seed)
    a, b, c = np.random.SeedSequence(runtime_seed).spawn(3)
    return tuple(int(s.generate_state(1)[0]) for s in (a, b, c))  # type: ignore[return-value]

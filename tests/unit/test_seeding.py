"""Tests for shared.seeding — seed_everything and derive_seeds."""

import random

import numpy as np
import pytest
import torch

from signal_extraction_rnn_lstm.shared.seeding import derive_seeds, seed_everything


def _draw_three() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), float(torch.rand(1).item())


def test_seed_everything_reproducible() -> None:
    seed_everything(42)
    a = _draw_three()
    seed_everything(42)
    b = _draw_three()
    assert a == b


def test_seed_everything_different_seeds_diverge() -> None:
    seed_everything(1)
    a = _draw_three()
    seed_everything(2)
    b = _draw_three()
    assert a != b


def test_seed_everything_rejects_none() -> None:
    with pytest.raises(TypeError):
        seed_everything(None)  # type: ignore[arg-type]


def test_derive_seeds_deterministic() -> None:
    assert derive_seeds(42) == derive_seeds(42)


def test_derive_seeds_distinct_for_distinct_input() -> None:
    assert derive_seeds(42) != derive_seeds(43)


def test_derive_seeds_returns_three_distinct_ints() -> None:
    cs, ss, ds = derive_seeds(42)
    assert len({cs, ss, ds}) == 3


def test_derive_seeds_rejects_none() -> None:
    with pytest.raises(TypeError):
        derive_seeds(None)  # type: ignore[arg-type]


def test_t_sd_04_deterministic_algorithms_enabled() -> None:
    """T-SD-04: seed_everything must enable PyTorch deterministic kernels.

    Pinned because PLAN.md § 11.1 / NFR-2 / PRD_training_evaluation § 5.3 all
    depend on this flag being on.  ``warn_only=True`` lets non-deterministic
    kernels fall back with a warning, which is the documented best-effort mode.
    """
    torch.use_deterministic_algorithms(False)
    assert torch.are_deterministic_algorithms_enabled() is False
    seed_everything(0)
    assert torch.are_deterministic_algorithms_enabled() is True

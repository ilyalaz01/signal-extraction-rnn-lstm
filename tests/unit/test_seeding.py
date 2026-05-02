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

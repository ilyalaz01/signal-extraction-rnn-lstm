"""Tests for services.models.base — T-MD-01, T-MD-02 + abstract-base contract."""

import pytest
import torch

from signal_extraction_rnn_lstm.services.models.base import (
    SignalExtractor,
    _to_fc_input,
    _to_seq_input,
)


def _selector(b: int) -> torch.Tensor:
    out = torch.zeros(b, 4, dtype=torch.float32)
    for i in range(b):
        out[i, i % 4] = 1.0
    return out


def test_t_md_01_to_fc_input_shape_dtype_values() -> None:
    sel = _selector(2)
    win = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    out = _to_fc_input(sel, win)
    assert out.shape == (2, 14)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, torch.cat([sel, win], dim=-1))


def test_t_md_02_to_seq_input_shape_dtype_values() -> None:
    sel = _selector(3)
    win = torch.arange(30, dtype=torch.float32).reshape(3, 10)
    out = _to_seq_input(sel, win)
    assert out.shape == (3, 10, 5)
    assert out.dtype == torch.float32
    # Per timestep t: out[b, t, 0] == w_noisy[b, t]; out[b, t, 1:5] == selector[b].
    for b in range(3):
        torch.testing.assert_close(out[b, :, 0], win[b])
        for t in range(10):
            torch.testing.assert_close(out[b, t, 1:5], sel[b])


def test_signal_extractor_is_abstract() -> None:
    with pytest.raises(TypeError):
        SignalExtractor()  # type: ignore[abstract]


def test_to_seq_input_handles_arbitrary_time_length() -> None:
    """The reshape helper is length-agnostic (PRD § 10 edge case)."""
    sel = _selector(2)
    win = torch.zeros(2, 7, dtype=torch.float32)
    assert _to_seq_input(sel, win).shape == (2, 7, 5)

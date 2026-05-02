"""Tests for services.models.rnn — T-MD-05, 08, 11, 14, 17."""

import pytest
import torch
from torch import nn

from signal_extraction_rnn_lstm.services.models.rnn import RNNConfig, RNNExtractor


def _onehot(b: int, k: int) -> torch.Tensor:
    sel = torch.zeros(b, 4, dtype=torch.float32)
    sel[:, k] = 1.0
    return sel


@pytest.fixture
def model() -> RNNExtractor:
    torch.manual_seed(0)
    return RNNExtractor(RNNConfig())


def test_t_md_05_forward_shape(model: RNNExtractor) -> None:
    for b in (1, 16, 64):
        out = model(_onehot(b, 0), torch.randn(b, 10))
        assert out.shape == (b, 10)
        assert out.dtype == torch.float32


def test_t_md_08_selector_responds(model: RNNExtractor) -> None:
    win = torch.randn(8, 10)
    assert (model(_onehot(8, 0), win) - model(_onehot(8, 2), win)).abs().max() > 1e-6


def test_t_md_11_trainability_overfit() -> None:
    """T-MD-11 (calibrated): see PROMPTS § 8 — step count bumped from 200 to 2000."""
    torch.manual_seed(0)
    model = RNNExtractor(RNNConfig())
    sel = torch.eye(4, dtype=torch.float32)
    win = torch.randn(4, 10)
    target = torch.randn(4, 10)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    final = float("nan")
    for _ in range(2000):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(sel, win), target)
        loss.backward()
        opt.step()
        final = loss.item()
    assert final < 5e-3, f"RNN failed to overfit 4 examples: final MSE = {final}"


def test_t_md_14_parameter_count(model: RNNExtractor) -> None:
    n = sum(p.numel() for p in model.parameters())
    assert 5_000 <= n <= 5_500, f"got {n}"


def test_t_md_17_rnn_uses_tanh(model: RNNExtractor) -> None:
    """PRD § 5.2: tanh is locked; ReLU would short-circuit the vanishing-grad regime."""
    assert model.rnn.nonlinearity == "tanh"


def test_rnn_config_validation() -> None:
    with pytest.raises(ValueError):
        RNNConfig(hidden=0)
    with pytest.raises(ValueError):
        RNNConfig(layers=0)

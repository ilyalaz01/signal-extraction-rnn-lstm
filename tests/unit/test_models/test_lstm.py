"""Tests for services.models.lstm — T-MD-06, 09, 12, 15, 18."""

import pytest
import torch
from torch import nn

from signal_extraction_rnn_lstm.services.models.lstm import LSTMConfig, LSTMExtractor


def _onehot(b: int, k: int) -> torch.Tensor:
    sel = torch.zeros(b, 4, dtype=torch.float32)
    sel[:, k] = 1.0
    return sel


@pytest.fixture
def model() -> LSTMExtractor:
    torch.manual_seed(0)
    return LSTMExtractor(LSTMConfig())


def test_t_md_06_forward_shape(model: LSTMExtractor) -> None:
    for b in (1, 16, 64):
        out = model(_onehot(b, 0), torch.randn(b, 10))
        assert out.shape == (b, 10)
        assert out.dtype == torch.float32


def test_t_md_09_selector_responds(model: LSTMExtractor) -> None:
    win = torch.randn(8, 10)
    assert (model(_onehot(8, 0), win) - model(_onehot(8, 2), win)).abs().max() > 1e-6


def test_t_md_12_trainability_overfit() -> None:
    """T-MD-12 (calibrated): see PROMPTS § 8 — LSTM SGD step count bumped to 5000.

    LSTM's gating means full-batch SGD lr=1e-2 converges noticeably slower than
    FC or RNN.  Empirically: 2000 steps stalls near 0.02; 5000 steps reaches
    ~3e-6.  Same algorithm and lr as the PRD; only the step budget changes.
    """
    torch.manual_seed(0)
    model = LSTMExtractor(LSTMConfig())
    sel = torch.eye(4, dtype=torch.float32)
    win = torch.randn(4, 10)
    target = torch.randn(4, 10)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    final = float("nan")
    for _ in range(5000):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(sel, win), target)
        loss.backward()
        opt.step()
        final = loss.item()
    assert final < 5e-3, f"LSTM failed to overfit 4 examples: final MSE = {final}"


def test_t_md_15_parameter_count(model: LSTMExtractor) -> None:
    n = sum(p.numel() for p in model.parameters())
    assert 18_200 <= n <= 18_900, f"got {n}"


def test_t_md_18_forget_gate_bias_not_jozefowicz(model: LSTMExtractor) -> None:
    """PRD § 6: confirm we are NOT applying the forget-bias=1.0 heuristic.

    PyTorch LSTM bias layout: bias_ih_l0 = (4*hidden,) in order (i, f, g, o).
    Forget-gate bias slice is [hidden:2*hidden]. Default init places it in
    [-sqrt(1/hidden), sqrt(1/hidden)] = [-0.125, 0.125] for hidden=64, so a
    threshold of 0.5 catches a Jozefowicz override (1.0) without flagging
    PyTorch defaults.
    """
    h = model.lstm.hidden_size
    forget_bias = model.lstm.bias_ih_l0[h:2 * h]
    assert forget_bias.abs().max().item() < 0.5


def test_lstm_config_validation() -> None:
    with pytest.raises(ValueError):
        LSTMConfig(hidden=0)
    with pytest.raises(ValueError):
        LSTMConfig(layers=0)

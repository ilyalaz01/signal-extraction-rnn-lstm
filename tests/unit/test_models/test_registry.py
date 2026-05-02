"""Cross-cutting model tests — T-MD-03 (registry), T-MD-19 (grad flow), T-MD-20 (sel diff)."""

import pytest
import torch
from torch import nn

from signal_extraction_rnn_lstm.services.models import (
    FCConfig,
    LSTMConfig,
    ModelConfig,
    RNNConfig,
    SignalExtractor,
    build,
    parse_model_config,
)
from signal_extraction_rnn_lstm.services.models.fc import FCExtractor
from signal_extraction_rnn_lstm.services.models.lstm import LSTMExtractor
from signal_extraction_rnn_lstm.services.models.rnn import RNNExtractor
from signal_extraction_rnn_lstm.shared.config import load_config

_KINDS = ("fc", "rnn", "lstm")


@pytest.fixture
def cfg() -> ModelConfig:
    return ModelConfig(fc=FCConfig(), rnn=RNNConfig(), lstm=LSTMConfig())


def test_t_md_03_dispatch_returns_correct_class(cfg: ModelConfig) -> None:
    assert isinstance(build("fc", cfg), FCExtractor)
    assert isinstance(build("rnn", cfg), RNNExtractor)
    assert isinstance(build("lstm", cfg), LSTMExtractor)
    for kind in _KINDS:
        assert isinstance(build(kind, cfg), SignalExtractor)


def test_t_md_03_unknown_kind_raises(cfg: ModelConfig) -> None:
    with pytest.raises(ValueError):
        build("transformer", cfg)  # type: ignore[arg-type]


def test_t_md_19_gradient_flow(cfg: ModelConfig) -> None:
    """For each model, backward fills a non-zero grad on every parameter tensor."""
    sel = torch.eye(4, dtype=torch.float32)
    win = torch.randn(4, 10)
    target = torch.randn(4, 10)
    for kind in _KINDS:
        torch.manual_seed(0)
        m = build(kind, cfg)
        loss = nn.functional.mse_loss(m(sel, win), target)
        loss.backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"{kind}.{name}: grad is None"
            assert p.grad.abs().sum().item() > 0, f"{kind}.{name}: grad is zero"


def test_t_md_20_selector_is_differentiable(cfg: ModelConfig) -> None:
    """Strengthens T-MD-07/08/09: not only does output depend on selector,
    the dependency is differentiable."""
    win = torch.randn(4, 10)
    for kind in _KINDS:
        torch.manual_seed(0)
        m = build(kind, cfg)
        sel = torch.eye(4, dtype=torch.float32, requires_grad=True)
        m(sel, win).sum().backward()
        assert sel.grad is not None, f"{kind}: sel.grad is None"
        assert sel.grad.abs().sum().item() > 0, f"{kind}: sel.grad is zero"


def test_parse_model_config_round_trip() -> None:
    cfg = load_config()
    mcfg = parse_model_config(cfg["model"])
    assert mcfg.fc.hidden == (64, 64)
    assert mcfg.rnn.hidden == 64 and mcfg.rnn.layers == 1
    assert mcfg.lstm.hidden == 64 and mcfg.lstm.layers == 1

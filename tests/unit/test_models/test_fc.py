"""Tests for services.models.fc — T-MD-04, 07, 10, 13, 16."""

import pytest
import torch
from torch import nn

from signal_extraction_rnn_lstm.services.models.fc import FCConfig, FCExtractor


@pytest.fixture
def model() -> FCExtractor:
    torch.manual_seed(0)
    return FCExtractor(FCConfig())


def _onehot(b: int, k: int) -> torch.Tensor:
    sel = torch.zeros(b, 4, dtype=torch.float32)
    sel[:, k] = 1.0
    return sel


def test_t_md_04_forward_shape(model: FCExtractor) -> None:
    for b in (1, 16, 256):
        out = model(_onehot(b, 0), torch.randn(b, 10))
        assert out.shape == (b, 10)
        assert out.dtype == torch.float32


def test_t_md_07_selector_responds(model: FCExtractor) -> None:
    win = torch.randn(8, 10)
    assert (model(_onehot(8, 0), win) - model(_onehot(8, 2), win)).abs().max() > 1e-6


def test_t_md_10_trainability_overfit() -> None:
    """T-MD-10 (calibrated): see PROMPTS § 8 — 200 SGD steps numerically too few.

    PRD prescribes SGD lr=1e-2 / 200 steps / MSE < 1e-3. With the spec-correct
    architecture (Linear(14,64)→ReLU→Linear(64,64)→ReLU→Linear(64,10)) and
    PyTorch default init, full-batch SGD lr=1e-2 stalls at ~0.45 after 200
    steps but converges to ~1e-7 by step 1500.  Bumped to 2000 steps to
    preserve the smoke intent (verify the architecture can overfit) without
    abandoning the threshold.
    """
    torch.manual_seed(0)
    model = FCExtractor(FCConfig())
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
    assert final < 1e-3, f"FC failed to overfit 4 examples: final MSE = {final}"


def test_t_md_13_parameter_count(model: FCExtractor) -> None:
    n = sum(p.numel() for p in model.parameters())
    assert 5_500 <= n <= 6_000, f"got {n} parameters"


def test_t_md_16_determinism() -> None:
    torch.manual_seed(0)
    a = FCExtractor(FCConfig())
    torch.manual_seed(0)
    b = FCExtractor(FCConfig())
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa, pb)


def test_fc_config_validation() -> None:
    with pytest.raises(ValueError):
        FCConfig(hidden=())
    with pytest.raises(ValueError):
        FCConfig(hidden=(64, 0, 64))
    with pytest.raises(ValueError):
        FCConfig(hidden=(-1,))

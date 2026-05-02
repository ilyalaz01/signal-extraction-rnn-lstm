"""Tests for shared.device.resolve_device."""

import pytest
import torch

from signal_extraction_rnn_lstm.shared.device import resolve_device


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_returns_known_device() -> None:
    d = resolve_device("auto")
    assert d.type in ("cpu", "cuda")


def test_resolve_device_cuda_when_available() -> None:
    if torch.cuda.is_available():  # pragma: no cover — env-dependent
        assert resolve_device("cuda") == torch.device("cuda")
    else:
        # torch.device('cuda') is constructible regardless of availability;
        # resolve_device just hands it back without checking availability.
        assert resolve_device("cuda") == torch.device("cuda")


def test_resolve_device_invalid_raises() -> None:
    with pytest.raises(ValueError):
        resolve_device("tpu")
    with pytest.raises(ValueError):
        resolve_device("")

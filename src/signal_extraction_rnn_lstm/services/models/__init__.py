"""Model registry — ``build(kind, config)`` is the only public surface.

Concrete model classes (FCExtractor, RNNExtractor, LSTMExtractor) are
internal to this package; the SDK calls ``build()`` and receives an opaque
``SignalExtractor``. See PRD_models.md § 7 and PLAN.md § 14 item 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from signal_extraction_rnn_lstm.services.models.base import SignalExtractor
from signal_extraction_rnn_lstm.services.models.fc import FCConfig, FCExtractor
from signal_extraction_rnn_lstm.services.models.lstm import LSTMConfig, LSTMExtractor
from signal_extraction_rnn_lstm.services.models.rnn import RNNConfig, RNNExtractor

ModelKind = Literal["fc", "rnn", "lstm"]
__all__ = [
    "FCConfig", "LSTMConfig", "ModelConfig", "ModelKind", "RNNConfig",
    "SignalExtractor", "build", "parse_model_config",
]


@dataclass(frozen=True)
class ModelConfig:
    """Aggregate config for the three architectures.

    Input:  fc / rnn / lstm sub-configs.
    Output: this object.  The training loop receives one ModelConfig and
            ``build(kind, cfg)`` picks the right slot.
    """

    fc: FCConfig
    rnn: RNNConfig
    lstm: LSTMConfig


_REGISTRY: dict[str, tuple[type[SignalExtractor], str]] = {
    "fc": (FCExtractor, "fc"),
    "rnn": (RNNExtractor, "rnn"),
    "lstm": (LSTMExtractor, "lstm"),
}


def build(kind: ModelKind, config: ModelConfig) -> SignalExtractor:
    """Instantiate a ``SignalExtractor`` for the given kind.

    Input:  kind ∈ {'fc', 'rnn', 'lstm'}, ModelConfig.
    Output: SignalExtractor instance with PyTorch default init.
    Raises: ValueError on unknown kind.
    """
    if kind not in _REGISTRY:
        raise ValueError(f"unknown model kind: {kind!r}; expected one of {list(_REGISTRY)}")
    cls, attr = _REGISTRY[kind]
    return cls(getattr(config, attr))


def parse_model_config(d: dict) -> ModelConfig:
    """Build a validated ``ModelConfig`` from ``config['model']`` dict."""
    return ModelConfig(
        fc=FCConfig(hidden=tuple(int(h) for h in d["fc"]["hidden"])),
        rnn=RNNConfig(hidden=int(d["rnn"]["hidden"]), layers=int(d["rnn"]["layers"])),
        lstm=LSTMConfig(hidden=int(d["lstm"]["hidden"]), layers=int(d["lstm"]["layers"])),
    )

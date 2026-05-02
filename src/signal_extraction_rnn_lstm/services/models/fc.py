"""Fully Connected model — non-temporal baseline (PRD_models.md § 5.1).

Input:  flat tensor (B, 14) = 4-dim one-hot selector ⊕ 10-dim noisy window
        (assembled internally via ``_to_fc_input``).
Output: tensor (B, 10) — predicted clean window for the selected sinusoid.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from signal_extraction_rnn_lstm.constants import FC_INPUT_SIZE, OUTPUT_SIZE
from signal_extraction_rnn_lstm.services.models.base import (
    SignalExtractor,
    _to_fc_input,
)


@dataclass(frozen=True)
class FCConfig:
    """Config for ``FCExtractor``.

    Input:  ``hidden`` — tuple of hidden-layer widths (≥ 1 entry, each > 0).
    Output: this object.
    Setup:  validation runs in ``__post_init__``.
    """

    hidden: tuple[int, ...] = (64, 64)

    def __post_init__(self) -> None:
        if not self.hidden:
            raise ValueError("FCConfig.hidden must have at least 1 entry")
        for h in self.hidden:
            if h <= 0:
                raise ValueError(f"FCConfig.hidden widths must be > 0, got {h}")


class FCExtractor(SignalExtractor):
    """Multi-layer fully connected baseline.

    Input:  selector (B, 4), w_noisy (B, 10).
    Output: w_pred (B, 10) float32.
    Setup:  ``Linear(14, h1) → ReLU → ... → Linear(h_n, 10)``; PyTorch
            default init (PRD_models § 6).
    """

    def __init__(self, config: FCConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = FC_INPUT_SIZE
        for h in config.hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, OUTPUT_SIZE))
        self.net = nn.Sequential(*layers)

    def forward(self, selector: torch.Tensor, w_noisy: torch.Tensor) -> torch.Tensor:
        return self.net(_to_fc_input(selector, w_noisy))

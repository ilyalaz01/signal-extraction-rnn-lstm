"""Vanilla RNN model — short-memory baseline (PRD_models § 5.2).

Input:  selector (B, 4), w_noisy (B, 10).  Internally reshaped to a
        sequence of length 10 with feature size 5 via ``_to_seq_input``.
Output: w_pred (B, 10) — sequence-to-vector head from the last timestep.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from signal_extraction_rnn_lstm.constants import OUTPUT_SIZE, SEQ_FEATURE_SIZE
from signal_extraction_rnn_lstm.services.models.base import (
    SignalExtractor,
    _to_seq_input,
)


@dataclass(frozen=True)
class RNNConfig:
    """Config for ``RNNExtractor``.

    Input:  ``hidden`` (int > 0) — recurrent state size; ``layers`` (int ≥ 1).
    Output: this object.
    Setup:  ``__post_init__`` enforces both constraints.
    """

    hidden: int = 64
    layers: int = 1

    def __post_init__(self) -> None:
        if self.hidden <= 0:
            raise ValueError(f"RNNConfig.hidden must be > 0, got {self.hidden}")
        if self.layers < 1:
            raise ValueError(f"RNNConfig.layers must be >= 1, got {self.layers}")


class RNNExtractor(SignalExtractor):
    """Vanilla RNN with tanh nonlinearity and a sequence-to-vector head.

    Input:  selector (B, 4), w_noisy (B, 10).
    Output: w_pred (B, 10) float32.
    Setup:  ``nn.RNN(input_size=5, hidden_size=H, nonlinearity='tanh')``;
            head reads the last-timestep output and projects to 10 dims.
            PyTorch default init (PRD_models § 6).
    """

    def __init__(self, config: RNNConfig) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=SEQ_FEATURE_SIZE,
            hidden_size=config.hidden,
            num_layers=config.layers,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.head = nn.Linear(config.hidden, OUTPUT_SIZE)

    def forward(self, selector: torch.Tensor, w_noisy: torch.Tensor) -> torch.Tensor:
        seq = _to_seq_input(selector, w_noisy)
        output, _ = self.rnn(seq)            # output: (B, T, H)
        return self.head(output[:, -1, :])

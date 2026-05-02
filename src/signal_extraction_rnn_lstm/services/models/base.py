"""Abstract base + selector-broadcast reshape utilities.

Defines the shared ``SignalExtractor(nn.Module, abc.ABC)`` interface that
FC, RNN, and LSTM all implement, plus the two private reshape helpers
(``_to_fc_input``, ``_to_seq_input``) that materialize the selector-broadcast
scheme (ADR-003).  All concrete models call these helpers — the reshape
lives once.
"""

from __future__ import annotations

import abc

import torch
from torch import nn


def _to_fc_input(selector: torch.Tensor, w_noisy: torch.Tensor) -> torch.Tensor:
    """Concatenate selector and window into a flat (B, 14) tensor.

    Input:  selector (B, 4), w_noisy (B, 10).
    Output: (B, 14) float32 — flat selector ⊕ window.
    Setup:  pure function. No allocation beyond ``torch.cat``.
    """
    return torch.cat([selector, w_noisy], dim=-1)


def _to_seq_input(selector: torch.Tensor, w_noisy: torch.Tensor) -> torch.Tensor:
    """Tile selector along time and concatenate per-step → (B, T, 5).

    Input:  selector (B, 4), w_noisy (B, T).
    Output: (B, T, 5) float32 — per step ``[w_noisy[t], C[0..3]]``.
    Setup:  pure function.  Selector is broadcast-tiled along the time axis.
    """
    sel_tiled = selector.unsqueeze(1).expand(-1, w_noisy.shape[-1], -1)
    w_unsq = w_noisy.unsqueeze(-1)
    return torch.cat([w_unsq, sel_tiled], dim=-1)


class SignalExtractor(nn.Module, abc.ABC):
    """Abstract base for FC, RNN, and LSTM extractors.

    Input:  ``(selector: (B, 4) float32, w_noisy: (B, 10) float32)``.
    Output: ``w_pred: (B, 10) float32``.
    Setup:  subclasses configure their own internal layers.  Models are
            invoked as ``model(selector, w_noisy)`` (PyTorch ``__call__``);
            no separate ``predict()`` method.
    """

    @abc.abstractmethod
    def forward(self, selector: torch.Tensor, w_noisy: torch.Tensor) -> torch.Tensor:
        ...

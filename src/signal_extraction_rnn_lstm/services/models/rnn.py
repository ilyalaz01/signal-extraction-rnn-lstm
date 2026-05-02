"""Vanilla RNN model.

Input:  sequence tensor (B, 10, 5) — per-step: 1 noisy sample ⊕ 4-dim
        broadcast selector (ADR-003 scheme).
Output: tensor (B, 10) — predicted clean window.

See PRD_models.md § 5.2 and HOMEWORK_BRIEF.md § 5.1.
"""

from signal_extraction_rnn_lstm.services.models.base import SignalExtractor


class RNNModel(SignalExtractor):
    """Vanilla recurrent network over a 10-step sequence.

    Input:  (B, SEQ_LEN, SEQ_FEATURE_SIZE) = (B, 10, 5).
    Output: (B, OUTPUT_SIZE)               = (B, 10).
    Setup:  config.model.rnn.hidden, config.model.rnn.layers.
    """

    def __init__(self, config: object) -> None:
        """Build RNN from config.model.rnn settings."""
        raise NotImplementedError("M3")

    def forward(self, x: object) -> object:
        """Forward pass through the RNN.

        Args:
            x: tensor of shape (B, 10, 5).

        Returns:
            Predicted window of shape (B, 10).
        """
        raise NotImplementedError("M3")

"""Fully Connected model — non-temporal baseline.

Input:  flat tensor (B, 14) = 4-dim one-hot selector ⊕ 10-dim noisy window.
Output: tensor (B, 10) — predicted clean window for the selected sinusoid.

See PRD_models.md § 5.1, HOMEWORK_BRIEF.md § 5.1, and ADR-003.
"""

from signal_extraction_rnn_lstm.services.models.base import SignalExtractor


class FCModel(SignalExtractor):
    """Multi-layer fully connected network.

    Input:  (B, FC_INPUT_SIZE) = (B, 14).
    Output: (B, OUTPUT_SIZE)   = (B, 10).
    Setup:  config.model.fc.hidden — list of hidden layer widths.
    """

    def __init__(self, config: object) -> None:
        """Build FC layers from config.model.fc.hidden."""
        raise NotImplementedError("M3")

    def forward(self, x: object) -> object:
        """Forward pass through stacked linear layers.

        Args:
            x: tensor of shape (B, 14).

        Returns:
            Predicted window of shape (B, 10).
        """
        raise NotImplementedError("M3")

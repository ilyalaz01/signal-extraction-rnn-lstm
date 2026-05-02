"""Base model interface.

Defines the shared forward-pass contract for FC, RNN, and LSTM.
All three models must produce output shape (B, 10) regardless of input
layout. See PRD_models.md § 5 and PLAN.md § 7.2.
"""


class SignalExtractor:
    """Abstract base for all three architectures.

    Public type contract fixed in PRD_models.md § 3.4. Subclasses must
    implement forward() with the shape contract from PLAN.md § 7.2.
    In M3 this class will extend torch.nn.Module.

    Input  (forward): model-specific — see subclass docstrings.
    Output (forward): tensor of shape (B, OUTPUT_SIZE) = (B, 10).
    """

    def forward(self, x: object) -> object:
        """Run the forward pass.

        Args:
            x: input tensor, shape depends on model kind.

        Returns:
            Predicted clean window, shape (B, 10).
        """
        raise NotImplementedError("M3")

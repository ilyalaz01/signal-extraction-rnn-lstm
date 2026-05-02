"""Evaluation service.

Computes overall MSE and per-frequency MSE breakdown on a dataset split.
See PRD_training_evaluation.md § 5 and HOMEWORK_BRIEF.md § 6.

Public surface:
    evaluate(model, dataloader, device) → EvalResult
"""


def evaluate(
    model: object,
    dataloader: object,
    device: object,
) -> object:
    """Evaluate model; return overall and per-frequency MSE.

    Args:
        model: trained model (SignalExtractor subclass, already on device).
        dataloader: a single DataLoader split (val or test).
        device: torch.device to move batches to.

    Returns:
        EvalResult(overall_mse, per_freq_mse) where per_freq_mse maps
        frequency index → scalar MSE.

    See PRD_training_evaluation.md § 5 for metric definitions.
    """
    raise NotImplementedError("M4")

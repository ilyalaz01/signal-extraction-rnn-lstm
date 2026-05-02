"""Training service.

Implements the training loop, optimizer wiring, checkpointing, and
per-epoch logging. See PRD_training_evaluation.md.

Public surface:
    train(model, datasets, config, run_dir) → TrainingResult
"""


def train(
    model: object,
    datasets: object,
    config: object,
    run_dir: object,
) -> object:
    """Train model for the configured number of epochs.

    Args:
        model: instantiated model (SignalExtractor subclass on the correct device).
        datasets: SplitDatasets produced by dataset.build_dataset().
        config: validated config dict (training section will be extracted).
        run_dir: Path to the run output directory for checkpoints and logs.

    Returns:
        TrainingResult(model_id, train_history, val_history, checkpoint_path).

    See PRD_training_evaluation.md § 4 for the training loop contract.
    """
    raise NotImplementedError("M4")

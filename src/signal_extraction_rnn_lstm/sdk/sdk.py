"""SDK layer — single entry point for all project operations.

External consumers (CLI scripts, notebooks, tests) import only this class.
No service module may be imported directly from outside the package.
See PLAN.md § 3–6 and SOFTWARE_PROJECT_GUIDELINES § 3.1.
"""

from pathlib import Path


class SDK:
    """Orchestrates signal generation, dataset construction, training, and evaluation.

    Input:
        config_path (Path | None): path to setup.json; defaults to
            <project_root>/config/setup.json.
    Output:
        SDK instance exposing all project operations.
    Setup:
        seed (int | None): overrides config ``runtime.seed``.
        device (str | None): 'cuda' | 'cpu' | 'auto'; overrides config.
    """

    def __init__(
        self,
        config_path: Path | None = None,
        *,
        seed: int | None = None,
        device: str | None = None,
    ) -> None:
        """Load config, seed RNG, and resolve compute device.

        Args:
            config_path: path to setup.json; None → default location.
            seed: seed override; None → use config value.
            device: device override; None → use config value.
        """
        raise NotImplementedError("M2")

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def generate_corpus(self) -> object:
        """Generate the 10-vector sinusoid corpus.

        Returns:
            Corpus dataclass (4 clean + 4 noisy sinusoids + two composite
            sums). See PRD_signal_generation.md for field contracts.
        """
        raise NotImplementedError("M2")

    def build_dataset(self, corpus: object) -> object:
        """Sample windows from corpus and return train/val/test splits.

        Args:
            corpus: Corpus produced by generate_corpus().

        Returns:
            SplitDatasets(train, val, test) — three DataLoader objects.
        """
        raise NotImplementedError("M2")

    def train(self, model_kind: str, datasets: object) -> object:
        """Train one model architecture on the provided datasets.

        Args:
            model_kind: one of 'fc', 'rnn', 'lstm'.
            datasets: SplitDatasets produced by build_dataset().

        Returns:
            TrainingResult(model_id, train_history, val_history,
                checkpoint_path).
        """
        raise NotImplementedError("M4")

    def evaluate(self, trained: object, datasets: object) -> object:
        """Evaluate a trained model; return overall and per-frequency MSE.

        Args:
            trained: TrainingResult (from train()) or a model_id string
                pointing to a saved checkpoint. MUST NOT be a raw nn.Module
                — that would leak internal types across the SDK boundary.
            datasets: SplitDatasets produced by build_dataset().

        Returns:
            EvalResult(overall_mse, per_freq_mse).
        """
        raise NotImplementedError("M4")

    def run_experiment(self, spec: object) -> object:
        """Run a complete experiment: generate → build → train → evaluate.

        Args:
            spec: ExperimentSpec describing model kind, config overrides,
                and output directory.

        Returns:
            ExperimentResult with all metrics and paths to artefacts.
        """
        raise NotImplementedError("M4")

"""SDK layer — single entry point for all project operations.

External consumers (CLI scripts, notebooks, tests) import only this class.
No service module may be imported directly from outside the package.
See PLAN.md § 3–6 and SOFTWARE_PROJECT_GUIDELINES § 3.1.
"""

from __future__ import annotations

from pathlib import Path

from signal_extraction_rnn_lstm.services.dataset import (
    SplitDatasets,
    build_split_datasets,
    parse_dataset_config,
)
from signal_extraction_rnn_lstm.services.signal_gen import (
    Corpus,
    generate_corpus,
    parse_signal_config,
)
from signal_extraction_rnn_lstm.shared.config import load_config
from signal_extraction_rnn_lstm.shared.device import resolve_device
from signal_extraction_rnn_lstm.shared.seeding import derive_seeds, seed_everything


class SDK:
    """Orchestrates signal generation, dataset construction, training, and evaluation.

    Input:
        config_path (Path | None): path to setup.json; defaults to
            ``<project_root>/config/setup.json``.
    Output:
        SDK instance exposing all project operations.
    Setup:
        seed (int | None): overrides ``config['runtime']['seed']``.
        device (str | None): ``'cuda' | 'cpu' | 'auto'``; overrides config.
    """

    def __init__(
        self,
        config_path: Path | None = None,
        *,
        seed: int | None = None,
        device: str | None = None,
    ) -> None:
        cfg = load_config(config_path)
        self._config = cfg
        self.seed: int = int(seed) if seed is not None else int(cfg["runtime"]["seed"])
        device_str = device if device is not None else cfg["runtime"]["device"]
        self.device = resolve_device(device_str)
        seed_everything(self.seed)
        self._corpus_seed, self._sampling_seed, self._dataloader_seed = derive_seeds(self.seed)
        self._signal_config = parse_signal_config(cfg["signal"])
        self._dataset_config = parse_dataset_config(cfg["dataset"])

    def generate_corpus(self) -> Corpus:
        """Generate the 10-vector sinusoid corpus."""
        return generate_corpus(self._signal_config, self._corpus_seed)

    def build_dataset(self, corpus: Corpus) -> SplitDatasets:
        """Sample windows from corpus and return train/val/test splits."""
        return build_split_datasets(corpus, self._dataset_config, self._sampling_seed)

    def train(self, model_kind: str, datasets: object) -> object:
        """Train one model architecture on the provided datasets (M4)."""
        raise NotImplementedError("M4")

    def evaluate(self, trained: object, datasets: object) -> object:
        """Evaluate a trained model; return overall and per-frequency MSE (M4)."""
        raise NotImplementedError("M4")

    def run_experiment(self, spec: object) -> object:
        """Run a complete experiment: generate → build → train → evaluate (M4)."""
        raise NotImplementedError("M4")

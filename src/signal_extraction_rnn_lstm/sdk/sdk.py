"""SDK layer — single entry point for all project operations.

External consumers (CLI scripts, notebooks, tests) import only this class.
No service module may be imported directly from outside the package.
See PLAN.md § 3–6 and SOFTWARE_PROJECT_GUIDELINES § 3.1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from signal_extraction_rnn_lstm.services.dataset import (
    SplitDatasets,
    build_split_datasets,
    parse_dataset_config,
)
from signal_extraction_rnn_lstm.services.evaluation import EvalResult
from signal_extraction_rnn_lstm.services.evaluation import evaluate as _evaluate_service
from signal_extraction_rnn_lstm.services.models import (
    ModelKind,
    build,
    parse_model_config,
)
from signal_extraction_rnn_lstm.services.signal_gen import (
    Corpus,
    generate_corpus,
    parse_signal_config,
)
from signal_extraction_rnn_lstm.services.training import (
    TrainingResult,
    parse_training_config,
)
from signal_extraction_rnn_lstm.services.training import train as _train_service
from signal_extraction_rnn_lstm.shared.config import apply_overrides, load_config
from signal_extraction_rnn_lstm.shared.device import resolve_device
from signal_extraction_rnn_lstm.shared.seeding import derive_seeds, seed_everything


@dataclass(frozen=True)
class ExperimentSpec:
    """Identifies one training run for ``SDK.run_experiment()``.

    Input:  model_kind, optional seed override, optional config overrides.
    Setup:  ``overrides`` is a flat dict of dotted-path → value pairs applied
            to setup.json before building the corpus.
    """

    model_kind: ModelKind
    seed: int | None = None
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentResult:
    """Output of ``SDK.run_experiment(spec)`` — spec + train + eval bundle."""

    spec: ExperimentSpec
    train_result: TrainingResult
    eval_result: EvalResult


class SDK:
    """Orchestrates signal generation, dataset construction, training, and evaluation."""

    def __init__(
        self,
        config_path: Path | None = None,
        *,
        seed: int | None = None,
        device: str | None = None,
        results_root: Path | None = None,
    ) -> None:
        cfg = load_config(config_path)
        self._config = cfg
        self.seed: int = int(seed) if seed is not None else int(cfg["runtime"]["seed"])
        self.device = resolve_device(device if device is not None else cfg["runtime"]["device"])
        seed_everything(self.seed)
        self._corpus_seed, self._sampling_seed, self._dataloader_seed = derive_seeds(self.seed)
        self._signal_config = parse_signal_config(cfg["signal"])
        self._dataset_config = parse_dataset_config(cfg["dataset"])
        if results_root is None:
            results_root = Path(__file__).resolve().parents[3] / "results"
        self._results_root = Path(results_root)
        self._results_root.mkdir(parents=True, exist_ok=True)

    def generate_corpus(self) -> Corpus:
        """Generate the 10-vector sinusoid corpus."""
        return generate_corpus(self._signal_config, self._corpus_seed)

    def build_dataset(self, corpus: Corpus) -> SplitDatasets:
        """Sample windows from corpus and return train/val/test splits."""
        return build_split_datasets(corpus, self._dataset_config, self._sampling_seed)

    def train(self, model_kind: ModelKind, datasets: SplitDatasets) -> TrainingResult:
        """Build a model of ``kind`` and train it on ``datasets``."""
        run_dir = self._make_run_dir(model_kind, self.seed)
        model = build(model_kind, parse_model_config(self._config["model"]))
        return _train_service(model, datasets,
                              parse_training_config(self._config["training"]),
                              run_dir, self._dataloader_seed)

    def evaluate(self, trained: TrainingResult, datasets: SplitDatasets) -> EvalResult:
        """Evaluate ``trained.model`` on ``datasets.test``; results.json written
        to ``trained.run_dir`` (without spec/training keys — call
        ``run_experiment`` if you need the full schema)."""
        return _evaluate_service(trained.model, datasets, trained.run_dir)

    def run_experiment(self, spec: ExperimentSpec) -> ExperimentResult:
        """Apply overrides, train, evaluate, and finalise results.json."""
        cfg = apply_overrides(self._config, spec.overrides)
        seed = int(spec.seed) if spec.seed is not None else int(cfg["runtime"]["seed"])
        seed_everything(seed)
        cs, ss, dls = derive_seeds(seed)
        sig_cfg = parse_signal_config(cfg["signal"])
        ds_cfg = parse_dataset_config(cfg["dataset"])
        tr_cfg = parse_training_config(cfg["training"])
        m_cfg = parse_model_config(cfg["model"])
        corpus = generate_corpus(sig_cfg, cs)
        splits = build_split_datasets(corpus, ds_cfg, ss)
        run_dir = self._make_run_dir(spec.model_kind, seed)
        model = build(spec.model_kind, m_cfg)
        tr = _train_service(model, splits, tr_cfg, run_dir, dls)
        ev = _evaluate_service(model, splits, run_dir)
        self._finalise_results_json(run_dir, spec, seed, tr, tr_cfg.epochs)
        result = ExperimentResult(spec=spec, train_result=tr, eval_result=ev)
        torch.save(result, run_dir / "result.pkl")  # ADR-007: rich persistence for analysis
        return result

    def run_grid(self, specs: list[ExperimentSpec]) -> list[ExperimentResult]:
        """Sequentially run a list of experiments; v1.00 has no parallelism."""
        return [self.run_experiment(s) for s in specs]

    def _make_run_dir(self, model_kind: str, seed: int) -> Path:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = self._results_root / f"{ts}__{model_kind}__{seed}"
        path.mkdir(parents=True, exist_ok=False)  # ADR-014 + PRD § 13: no silent overwrite
        return path

    def _finalise_results_json(self, run_dir: Path, spec: ExperimentSpec, seed: int,
                                tr: TrainingResult, configured_epochs: int) -> None:
        path = run_dir / "results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["spec"] = {"model_kind": spec.model_kind, "seed": seed,
                           "overrides": dict(spec.overrides)}
        payload["training"] = {"best_epoch": tr.best_epoch, "best_val_mse": tr.best_val_mse,
                               "epochs_run": len(tr.train_history),
                               "stopped_early": len(tr.train_history) < configured_epochs}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

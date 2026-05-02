"""Evaluation service.

Computes overall test MSE and per-frequency MSE breakdown per
PRD_training_evaluation § 7, and writes ``results.json`` to ``run_dir``
with the schema described in § 10.4.  ``spec`` and ``training`` top-level
keys are written as ``{}`` here and filled by ``SDK.run_experiment``.

Public surface:
    EvalResult                  (frozen dataclass)
    evaluate(model, datasets, run_dir) → EvalResult
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from signal_extraction_rnn_lstm.constants import N_SINUSOIDS
from signal_extraction_rnn_lstm.services.dataset import SplitDatasets
from signal_extraction_rnn_lstm.services.models.base import SignalExtractor

_RESULTS_VERSION = "1.00"
_EVAL_BATCH_SIZE = 256


@dataclass(frozen=True)
class EvalResult:
    """Overall + per-frequency MSE on the test split.

    ``frequencies_hz`` is captured at evaluate-time from
    ``datasets.test.corpus.frequencies_hz`` so the ``per_freq_hz`` mapping
    survives any later ablation that changes the locked frequency set.
    """

    overall_test_mse: float
    per_freq_mse: dict[int, float]
    run_dir: Path
    frequencies_hz: tuple[float, ...] = field(default=())

    @property
    def per_freq_hz(self) -> dict[float, float]:
        """Map ``frequencies_hz[k]`` → ``per_freq_mse[k]`` for notebook use."""
        return {self.frequencies_hz[k]: self.per_freq_mse[k]
                for k in self.per_freq_mse}


def evaluate(model: SignalExtractor, datasets: SplitDatasets, run_dir: Path) -> EvalResult:
    """Evaluate ``model`` on ``datasets.test`` and write ``results.json``.

    Sets the model to ``eval()`` mode for the computation and restores its
    prior ``training`` flag on exit (PRD § 13).  ``no_grad`` context is used
    around the forward pass.
    """
    was_training = model.training
    model.eval()
    try:
        test_dl = DataLoader(datasets.test, batch_size=_EVAL_BATCH_SIZE,
                             shuffle=False, num_workers=0)
        preds, targets, ks = [], [], []
        with torch.no_grad():
            for batch in test_dl:
                preds.append(model(batch.selector, batch.w_noisy))
                targets.append(batch.w_clean)
                ks.append(batch.selector.argmax(dim=-1))
        pred = torch.cat(preds)
        target = torch.cat(targets)
        k_all = torch.cat(ks)
        overall_mse = float(nn.functional.mse_loss(pred, target, reduction="mean"))
        per_freq_mse: dict[int, float] = {}
        for ki in range(N_SINUSOIDS):
            mask = k_all == ki
            if mask.sum().item() == 0:
                per_freq_mse[ki] = float("nan")
            else:
                per_freq_mse[ki] = float(
                    nn.functional.mse_loss(pred[mask], target[mask], reduction="mean")
                )
    finally:
        model.train(was_training)
    frequencies_hz = tuple(datasets.test.corpus.frequencies_hz)
    result = EvalResult(
        overall_test_mse=overall_mse,
        per_freq_mse=per_freq_mse,
        run_dir=run_dir,
        frequencies_hz=frequencies_hz,
    )
    _write_results_json(run_dir, result)
    return result


def _write_results_json(run_dir: Path, result: EvalResult) -> None:
    """Write ``results.json`` with placeholder spec/training (filled by SDK)."""
    payload = {
        "version": _RESULTS_VERSION,
        "spec": {},
        "training": {},
        "evaluation": {
            "overall_test_mse": result.overall_test_mse,
            "per_freq_mse": {str(k): v for k, v in result.per_freq_mse.items()},
        },
    }
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

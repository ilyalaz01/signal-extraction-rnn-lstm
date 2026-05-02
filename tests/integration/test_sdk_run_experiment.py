"""Integration tests for SDK.run_experiment / run_grid — T-IT-01, T-IT-03.

T-IT-02 (reproducibility) lives in test_reproducibility.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from signal_extraction_rnn_lstm.sdk import SDK
from signal_extraction_rnn_lstm.sdk.sdk import ExperimentResult, ExperimentSpec

_TINY_OVERRIDES: dict[str, object] = {
    "dataset.n_train": 200, "dataset.n_val": 50, "dataset.n_test": 50,
    "training.epochs": 1, "training.batch_size": 64, "training.early_stop_patience": 0,
}


@pytest.mark.parametrize("kind", ["fc", "rnn", "lstm"])
def test_t_it_01_end_to_end_smoke(kind: str, tmp_path: Path) -> None:
    sdk = SDK(seed=0, device="cpu", results_root=tmp_path)
    spec = ExperimentSpec(model_kind=kind, seed=0, overrides=_TINY_OVERRIDES)
    result = sdk.run_experiment(spec)

    assert isinstance(result, ExperimentResult)
    assert result.spec.model_kind == kind
    assert result.eval_result.overall_test_mse > 0
    assert set(result.eval_result.per_freq_mse.keys()) == {0, 1, 2, 3}
    rd = result.train_result.run_dir
    for f in ("checkpoint_best.pt", "checkpoint_final.pt", "train.log",
              "results.json", "result.pkl"):
        assert (rd / f).exists(), f"missing {f}"

    payload = json.loads((rd / "results.json").read_text(encoding="utf-8"))
    assert payload["spec"]["model_kind"] == kind
    assert payload["training"]["best_epoch"] == result.train_result.best_epoch
    assert set(payload["evaluation"]["per_freq_mse"].keys()) == {"0", "1", "2", "3"}


def test_t_it_03_run_grid_distinct_run_dirs(tmp_path: Path) -> None:
    sdk = SDK(seed=0, device="cpu", results_root=tmp_path)
    specs = [ExperimentSpec(model_kind=k, seed=0, overrides=_TINY_OVERRIDES)
             for k in ("fc", "rnn", "lstm")]
    results = sdk.run_grid(specs)
    assert len(results) == 3
    run_dirs = [r.train_result.run_dir for r in results]
    assert len({rd.name for rd in run_dirs}) == 3
    for rd in run_dirs:
        assert rd.exists() and (rd / "results.json").exists()


def test_run_experiment_propagates_override_to_corpus(tmp_path: Path) -> None:
    """T-TR-09 part 2 — apply_overrides reaches the SignalConfig."""
    sdk = SDK(seed=0, device="cpu", results_root=tmp_path)
    spec = ExperimentSpec(
        model_kind="fc", seed=0,
        overrides={**_TINY_OVERRIDES, "signal.noise.alpha": 0.20},
    )
    result = sdk.run_experiment(spec)
    payload = json.loads((result.train_result.run_dir / "results.json").read_text())
    assert payload["spec"]["overrides"]["signal.noise.alpha"] == 0.20

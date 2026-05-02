"""T-IT-02: same spec + same seed → identical train_history and overall_test_mse."""

from __future__ import annotations

import math
from pathlib import Path

from signal_extraction_rnn_lstm.sdk import SDK
from signal_extraction_rnn_lstm.sdk.sdk import ExperimentSpec

_TINY = {
    "dataset.n_train": 200, "dataset.n_val": 50, "dataset.n_test": 50,
    "training.epochs": 2, "training.batch_size": 64, "training.early_stop_patience": 0,
}


def test_t_it_02_reproducibility_same_seed(tmp_path: Path) -> None:
    spec = ExperimentSpec(model_kind="fc", seed=7, overrides=_TINY)
    a = SDK(seed=7, device="cpu", results_root=tmp_path / "a").run_experiment(spec)
    b = SDK(seed=7, device="cpu", results_root=tmp_path / "b").run_experiment(spec)

    assert len(a.train_result.train_history) == len(b.train_result.train_history)
    for ea, eb in zip(a.train_result.train_history, b.train_result.train_history, strict=True):
        assert math.isclose(ea.train_mse, eb.train_mse, rel_tol=1e-7, abs_tol=1e-9)
        assert math.isclose(ea.val_mse, eb.val_mse, rel_tol=1e-7, abs_tol=1e-9)

    assert math.isclose(a.eval_result.overall_test_mse, b.eval_result.overall_test_mse,
                        rel_tol=1e-7, abs_tol=1e-9)

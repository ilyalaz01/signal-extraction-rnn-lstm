"""T-IT-02: same spec + same seed → bit-identical state_dict + identical metrics.

The bit-identical state_dict assertion (per audit AUDIT-2026-05 fix A1) is
strictly stronger than MSE-tolerance equality and only holds when
``torch.use_deterministic_algorithms(True, warn_only=True)`` is active —
which ``shared.seeding.seed_everything`` now enables (T-SD-04 in
``tests/unit/test_seeding.py``).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

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

    # Stronger: best-checkpoint weights must be bit-identical.
    sa = a.train_result.model.state_dict()
    sb = b.train_result.model.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"weights diverge at {k!r}"

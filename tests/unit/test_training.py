"""Tests for services.training — T-TR-01..08, T-TR-11.

T-TR-09 (apply_overrides) is in tests/unit/test_config.py.
T-TR-10 (run_dir naming) is an SDK responsibility, tested in M4c.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from signal_extraction_rnn_lstm.services.dataset import (
    DatasetConfig,
    SplitDatasets,
    build_split_datasets,
)
from signal_extraction_rnn_lstm.services.models.fc import FCConfig, FCExtractor
from signal_extraction_rnn_lstm.services.signal_gen import (
    Corpus,
    SignalConfig,
    generate_corpus,
)
from signal_extraction_rnn_lstm.services.training import (
    TrainingConfig,
    TrainingResult,
    _early_stop_index,
    parse_training_config,
    train,
)
from signal_extraction_rnn_lstm.shared.config import load_config


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return generate_corpus(SignalConfig(
        fs=1000, duration_s=10, frequencies_hz=(2.0, 10.0, 50.0, 200.0),
        amplitudes=(1.0, 1.0, 1.0, 1.0),
        phases_rad=(0.0, math.pi / 2, math.pi, 3 * math.pi / 2),
        noise_alpha=0.05, noise_beta=2 * math.pi, noise_distribution="gaussian",
    ), seed=42)


@pytest.fixture
def small_splits(corpus: Corpus) -> SplitDatasets:
    return build_split_datasets(
        corpus, DatasetConfig(window=10, n_train=200, n_val=50, n_test=50), sampling_seed=7,
    )


def _fresh_model() -> FCExtractor:
    torch.manual_seed(0)
    return FCExtractor(FCConfig())


def _train_cfg(**kw: object) -> TrainingConfig:
    base = {"batch_size": 64, "epochs": 3, "early_stop_patience": 5,
            "optimizer": "adam", "lr": 1e-3, "scheduler": None}
    base.update(kw)  # type: ignore[arg-type]
    return TrainingConfig(**base)  # type: ignore[arg-type]


def test_t_tr_01_loss_uses_mean_reduction() -> None:
    crit = nn.MSELoss(reduction="mean")
    y = torch.tensor([[1.0, 2.0, 3.0]])
    t = torch.tensor([[0.0, 0.0, 0.0]])
    manual = float(((y - t) ** 2).mean())
    assert abs(float(crit(y, t)) - manual) < 1e-6


def test_t_tr_02_loss_decreases_substantially(corpus: Corpus, tmp_path: Path) -> None:
    """PRD T-TR-02: 50 epochs full-batch on a 4-example mini-set; final ≤ 50% initial.

    Implementation note: the PRD writes "full-batch SGD" but the training service
    is Adam-only by design (PRD § 4.1).  Using Adam (the actual configured
    optimizer); the smoke intent — gross arch bugs flagged by failure to drop —
    is preserved.  ``train_history[0].train_mse`` is the pre-step loss for the
    first batch, which equals "initial MSE" when batch_size == full dataset.
    """
    splits = build_split_datasets(
        corpus, DatasetConfig(window=10, n_train=4, n_val=2, n_test=2), sampling_seed=0,
    )
    cfg = _train_cfg(batch_size=4, epochs=50, early_stop_patience=0)
    res = train(_fresh_model(), splits, cfg, tmp_path, dataloader_seed=0)
    initial = res.train_history[0].train_mse
    final = res.train_history[-1].train_mse
    assert final < 0.5 * initial, f"final={final}, initial={initial}"


def test_t_tr_03_early_stop_helper_logic() -> None:
    # Best at epoch 0; 5 stagnant epochs follow; with patience=3 → stop at idx 3.
    assert _early_stop_index([1.0, 1.5, 1.5, 1.5, 1.5, 1.5], patience=3) == 3
    # Best at epoch 2; flat after; patience=2 → stop at epoch 4.
    assert _early_stop_index([3.0, 2.5, 2.0, 2.5, 2.5], patience=2) == 4
    # Patience=0 disables early stopping.
    assert _early_stop_index([1.0, 2.0, 3.0], patience=0) is None
    # Always-improving sequence never triggers.
    assert _early_stop_index([5.0, 4.0, 3.0, 2.0], patience=2) is None


def test_t_tr_04_05_06_07_checkpoints_history_log(small_splits: SplitDatasets,
                                                   tmp_path: Path) -> None:
    model = _fresh_model()
    cfg = _train_cfg(epochs=4)
    res = train(model, small_splits, cfg, tmp_path, dataloader_seed=0)

    # T-TR-06 invariants
    assert isinstance(res, TrainingResult)
    assert 0 <= res.best_epoch < len(res.train_history)
    assert res.best_val_mse == res.train_history[res.best_epoch].val_mse
    assert (tmp_path / "train.log").exists()
    assert (tmp_path / "checkpoint_best.pt").exists()

    # T-TR-04: best checkpoint metadata reflects the best epoch
    best = torch.load(tmp_path / "checkpoint_best.pt", weights_only=True)
    assert best["epoch"] == res.best_epoch
    assert math.isclose(best["val_mse"], res.best_val_mse, abs_tol=1e-9)

    # T-TR-05: best weights are restored into TrainingResult.model
    for k, v in res.model.state_dict().items():
        torch.testing.assert_close(v, best["model_state_dict"][k])

    # T-TR-07: train.log layout
    rows = (tmp_path / "train.log").read_text().splitlines()
    assert rows[0] == "epoch\ttrain_mse\tval_mse\telapsed_s"
    assert len(rows) == 1 + len(res.train_history)
    for i, row in enumerate(rows[1:]):
        cols = row.split("\t")
        assert int(cols[0]) == i
        assert float(cols[1]) > 0 and float(cols[2]) > 0


def test_t_tr_08_determinism(small_splits: SplitDatasets, tmp_path: Path) -> None:
    a = train(_fresh_model(), small_splits, _train_cfg(epochs=3),
              tmp_path / "a", dataloader_seed=0) if (tmp_path / "a").mkdir() or True else None
    b = train(_fresh_model(), small_splits, _train_cfg(epochs=3),
              tmp_path / "b", dataloader_seed=0) if (tmp_path / "b").mkdir() or True else None
    assert a is not None and b is not None
    for ea, eb in zip(a.train_history, b.train_history, strict=True):
        assert math.isclose(ea.train_mse, eb.train_mse, rel_tol=1e-6)
        assert math.isclose(ea.val_mse, eb.val_mse, rel_tol=1e-6)


def test_t_tr_11_final_checkpoint_holds_final_weights(small_splits: SplitDatasets,
                                                       tmp_path: Path) -> None:
    """PRD § 4.3 fix: checkpoint_final.pt must hold final-epoch weights, not
    the best-epoch weights that get loaded back into ``model`` at return."""
    model = _fresh_model()
    cfg = _train_cfg(epochs=8, early_stop_patience=0)  # no early stop, full 8 epochs
    res = train(model, small_splits, cfg, tmp_path, dataloader_seed=0)

    if res.best_epoch == res.train_history[-1].epoch:
        pytest.skip("best == final this run; T-TR-11 invariant requires divergence")
    best = torch.load(tmp_path / "checkpoint_best.pt", weights_only=True)
    final = torch.load(tmp_path / "checkpoint_final.pt", weights_only=True)
    assert final["epoch"] == res.train_history[-1].epoch
    assert any(
        not torch.equal(best["model_state_dict"][k], final["model_state_dict"][k])
        for k in best["model_state_dict"]
    ), "checkpoint_final must differ from checkpoint_best when best != final"


@pytest.mark.parametrize("kw", [
    {"batch_size": 0}, {"epochs": 0}, {"optimizer": "sgd"},
    {"lr": 0.0}, {"scheduler": "cosine"}, {"early_stop_patience": -1},
])
def test_training_config_validation(kw: dict) -> None:
    with pytest.raises(ValueError):
        _train_cfg(**kw)


def test_parse_training_config_round_trip() -> None:
    cfg = parse_training_config(load_config()["training"])
    assert cfg.batch_size == 256 and cfg.epochs == 30
    assert cfg.optimizer == "adam" and cfg.scheduler is None


def test_kind_from_model_rejects_unknown_subclass() -> None:
    """Pin _kind_from_model raise path for non-{FC,RNN,LSTM} extractors."""
    from signal_extraction_rnn_lstm.services.models.base import SignalExtractor
    from signal_extraction_rnn_lstm.services.training import _kind_from_model

    class CustomExtractor(SignalExtractor):
        def forward(self, sel: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
            return torch.zeros(sel.shape[0], 10)

    with pytest.raises(ValueError, match="unknown"):
        _kind_from_model(CustomExtractor())


def test_train_raises_on_nan_loss(small_splits: SplitDatasets, tmp_path: Path) -> None:
    """Training aborts cleanly on NaN/Inf loss (PRD § 13)."""
    from signal_extraction_rnn_lstm.services.models.base import SignalExtractor

    class NanModel(SignalExtractor):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.tensor([1.0]))

        def forward(self, sel: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
            return torch.full((sel.shape[0], 10), float("nan")) * self.w

    with pytest.raises(ValueError, match="NaN"):
        train(NanModel(), small_splits, _train_cfg(epochs=1), tmp_path, dataloader_seed=0)


def test_train_early_stop_break_fires(small_splits: SplitDatasets, tmp_path: Path) -> None:
    """Lr=1.0 with Adam destabilises training; patience=2 stops it early."""
    cfg = _train_cfg(batch_size=64, epochs=30, early_stop_patience=2, lr=1.0)
    res = train(_fresh_model(), small_splits, cfg, tmp_path, dataloader_seed=0)
    assert len(res.train_history) < 30, f"early stop didn't fire ({len(res.train_history)} epochs)"

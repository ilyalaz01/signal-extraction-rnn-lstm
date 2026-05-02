"""Training service.

Implements PRD_training_evaluation.md § 2–6: MSE loss with reduction='mean',
Adam optimizer, early stopping on val MSE with best-checkpoint restoration,
and the ADR-014 run-dir layout (checkpoint_best.pt, checkpoint_final.pt,
train.log).

Public surface:
    TrainingConfig          (frozen dataclass — parsed config['training'])
    EpochResult             (mutable dataclass — one row in train_history)
    TrainingResult          (mutable — carries model with best-checkpoint weights)
    parse_training_config(d)→ TrainingConfig
    train(model, datasets, config, run_dir, dataloader_seed) → TrainingResult
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from signal_extraction_rnn_lstm.services.dataset import SplitDatasets
from signal_extraction_rnn_lstm.services.models.base import SignalExtractor

_KIND_BY_CLASS = {"FCExtractor": "fc", "RNNExtractor": "rnn", "LSTMExtractor": "lstm"}


@dataclass(frozen=True)
class TrainingConfig:
    """Parsed and validated form of ``config['training']``.

    Setup:  ``__post_init__`` enforces v1.00 constraints (Adam only,
            scheduler null only).  See PRD § 4.1.
    """

    batch_size: int
    epochs: int
    early_stop_patience: int
    optimizer: str
    lr: float
    scheduler: str | None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.early_stop_patience < 0:
            raise ValueError(f"early_stop_patience must be >= 0, got {self.early_stop_patience}")
        if self.optimizer != "adam":
            raise ValueError(f"only 'adam' supported in v1.00, got {self.optimizer!r}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.scheduler is not None:
            raise ValueError(f"only null scheduler supported in v1.00, got {self.scheduler!r}")


@dataclass
class EpochResult:
    """One row in train_history / train.log."""

    epoch: int
    train_mse: float
    val_mse: float
    elapsed_s: float


@dataclass
class TrainingResult:
    """Carries the best-checkpoint model + epoch history.  Not frozen because
    nn.Module is mutable."""

    model: SignalExtractor
    model_kind: str
    train_history: list[EpochResult] = field(default_factory=list)
    best_epoch: int = 0
    best_val_mse: float = float("inf")
    run_dir: Path = field(default_factory=Path)


def parse_training_config(d: dict) -> TrainingConfig:
    """Build a validated ``TrainingConfig`` from ``config['training']`` dict."""
    return TrainingConfig(
        batch_size=int(d["batch_size"]),
        epochs=int(d["epochs"]),
        early_stop_patience=int(d["early_stop_patience"]),
        optimizer=str(d["optimizer"]),
        lr=float(d["lr"]),
        scheduler=d.get("scheduler"),
    )


def _early_stop_index(val_mses: list[float], patience: int) -> int | None:
    """Return the epoch index at which training should stop (post-epoch check),
    or None if it never stops early.  Stops when val MSE has not strictly
    improved for ``patience`` consecutive epochs after the running best.
    ``patience <= 0`` disables.
    """
    if patience <= 0:
        return None
    best_val, best_i = float("inf"), -1
    for i, v in enumerate(val_mses):
        if v < best_val:
            best_val, best_i = v, i
        elif i - best_i >= patience:
            return i
    return None


def _kind_from_model(model: SignalExtractor) -> str:
    name = type(model).__name__
    if name not in _KIND_BY_CLASS:
        raise ValueError(f"unknown SignalExtractor subclass: {name}")
    return _KIND_BY_CLASS[name]


def _run_epoch(model: SignalExtractor, dl: DataLoader, criterion: nn.Module,
               optimizer: torch.optim.Optimizer | None) -> float:
    """One pass through ``dl``; ``optimizer is None`` means eval mode."""
    is_train = optimizer is not None
    model.train(is_train)
    total, n = 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in dl:
            if is_train:
                optimizer.zero_grad()
            loss = criterion(model(batch.selector, batch.w_noisy), batch.w_clean)
            if is_train:
                if not torch.isfinite(loss):
                    raise ValueError(f"NaN/Inf loss; check init/lr (got {loss.item()})")
                loss.backward()
                optimizer.step()
            total += loss.item()
            n += 1
    return total / max(n, 1)


def train(model: SignalExtractor, datasets: SplitDatasets, config: TrainingConfig,
          run_dir: Path, dataloader_seed: int) -> TrainingResult:
    """Train ``model`` on ``datasets.train`` with early stopping on val MSE.

    Writes ``checkpoint_best.pt``, ``checkpoint_final.pt``, and ``train.log``
    to ``run_dir`` (must exist).  On return, ``model`` has the best-checkpoint
    weights loaded back (T-TR-05).
    """
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0)
    train_gen = torch.Generator().manual_seed(dataloader_seed)
    train_dl = DataLoader(datasets.train, batch_size=config.batch_size, shuffle=True,
                          generator=train_gen, num_workers=0)
    val_dl = DataLoader(datasets.val, batch_size=config.batch_size, shuffle=False, num_workers=0)
    history: list[EpochResult] = []
    best_val, best_epoch, patience_ctr = float("inf"), 0, 0
    with (run_dir / "train.log").open("w", encoding="utf-8") as log_f:
        log_f.write("epoch\ttrain_mse\tval_mse\telapsed_s\n")
        log_f.flush()
        for epoch in range(config.epochs):
            t0 = time.time()
            train_mse = _run_epoch(model, train_dl, criterion, optimizer)
            val_mse = _run_epoch(model, val_dl, criterion, None)
            elapsed = time.time() - t0
            history.append(EpochResult(epoch, train_mse, val_mse, elapsed))
            log_f.write(f"{epoch}\t{train_mse:.6f}\t{val_mse:.6f}\t{elapsed:.3f}\n")
            log_f.flush()
            if val_mse < best_val:
                best_val, best_epoch, patience_ctr = val_mse, epoch, 0
                torch.save({"epoch": epoch, "val_mse": val_mse,
                            "model_state_dict": model.state_dict()},
                           run_dir / "checkpoint_best.pt")
            else:
                patience_ctr += 1
                if config.early_stop_patience > 0 and patience_ctr >= config.early_stop_patience:
                    break
    # Capture FINAL-epoch weights BEFORE restoring best (PRD § 4.3 fix; T-TR-11).
    last = history[-1]
    final_state = {k: v.clone() for k, v in model.state_dict().items()}
    torch.save({"epoch": last.epoch, "val_mse": last.val_mse,
                "model_state_dict": final_state}, run_dir / "checkpoint_final.pt")
    best_payload = torch.load(run_dir / "checkpoint_best.pt", weights_only=True)
    model.load_state_dict(best_payload["model_state_dict"])
    return TrainingResult(model=model, model_kind=_kind_from_model(model),
                          train_history=history, best_epoch=best_epoch,
                          best_val_mse=best_val, run_dir=run_dir)

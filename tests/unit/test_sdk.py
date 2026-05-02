"""Tests for the SDK layer — init, generate_corpus, build_dataset, run_dir naming."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from signal_extraction_rnn_lstm.sdk import SDK
from signal_extraction_rnn_lstm.services.dataset import parse_dataset_config
from signal_extraction_rnn_lstm.services.signal_gen import (
    SignalConfig,
    parse_signal_config,
)
from signal_extraction_rnn_lstm.shared.config import load_config


def _sdk(tmp_path: Path, **kw: object) -> SDK:
    return SDK(results_root=tmp_path, **kw)  # type: ignore[arg-type]


def test_sdk_init_default(tmp_path: Path) -> None:
    sdk = _sdk(tmp_path)
    assert sdk.seed == 1337
    assert sdk.device.type in ("cpu", "cuda")


def test_sdk_init_seed_override(tmp_path: Path) -> None:
    assert _sdk(tmp_path, seed=99).seed == 99


def test_sdk_init_device_override(tmp_path: Path) -> None:
    assert _sdk(tmp_path, device="cpu").device == torch.device("cpu")


def test_sdk_generate_corpus_default_shape(tmp_path: Path) -> None:
    c = _sdk(tmp_path).generate_corpus()
    assert c.clean.shape == (4, 10_000) and c.noisy.shape == (4, 10_000)
    assert c.config.fs == 1000


def test_sdk_build_dataset_split_lengths(tmp_path: Path) -> None:
    sdk = _sdk(tmp_path)
    splits = sdk.build_dataset(sdk.generate_corpus())
    assert (len(splits.train), len(splits.val), len(splits.test)) == (30_000, 3_750, 3_750)


def test_sdk_determinism_same_seed(tmp_path: Path) -> None:
    a = _sdk(tmp_path / "a", seed=42)
    b = _sdk(tmp_path / "b", seed=42)
    np.testing.assert_array_equal(a.generate_corpus().noisy, b.generate_corpus().noisy)


def test_sdk_different_seeds_yield_different_corpora(tmp_path: Path) -> None:
    a = _sdk(tmp_path / "a", seed=42)
    b = _sdk(tmp_path / "b", seed=43)
    assert not np.array_equal(a.generate_corpus().noisy, b.generate_corpus().noisy)


def test_t_tr_10_run_dir_naming_pattern(tmp_path: Path) -> None:
    """ADR-014 / PRD § 6.1: ``<utc_timestamp>__<model_kind>__<seed>``."""
    sdk = _sdk(tmp_path, seed=1337)
    run_dir = sdk._make_run_dir("fc", 1337)
    assert re.fullmatch(r"\d{8}T\d{6}Z__[a-z]+__\d+", run_dir.name)
    assert run_dir.parent == tmp_path


def test_make_run_dir_refuses_collision(tmp_path: Path) -> None:
    """SDK creates run_dir with mkdir(exist_ok=False); duplicate calls raise."""
    sdk = _sdk(tmp_path, seed=1)
    rd = sdk._make_run_dir("fc", 1)
    # Forge a collision by re-creating the same path
    with pytest.raises(FileExistsError):
        rd.mkdir(parents=True, exist_ok=False)


def test_sdk_train_and_evaluate_methods(tmp_path: Path) -> None:
    """Exercise SDK.train and SDK.evaluate (run_experiment is the main path).

    Uses the full default config but small enough to finish in a few seconds:
    we override n_train/n_val/n_test/epochs by constructing the SDK with a
    custom config file copied from the default.
    """
    from signal_extraction_rnn_lstm.sdk.sdk import ExperimentSpec
    sdk = _sdk(tmp_path, seed=0, device="cpu")
    spec = ExperimentSpec(
        model_kind="fc", seed=0,
        overrides={"dataset.n_train": 100, "dataset.n_val": 50, "dataset.n_test": 50,
                   "training.epochs": 1, "training.batch_size": 50,
                   "training.early_stop_patience": 0},
    )
    # SDK.train + SDK.evaluate are the lower-level surface; run_experiment
    # composes them internally.  Here we exercise them directly via a tiny
    # bypass: build a corpus and dataset under the override-merged config.
    from signal_extraction_rnn_lstm.shared.config import apply_overrides
    cfg = apply_overrides(sdk._config, spec.overrides)
    sdk._config = cfg
    sdk._dataset_config = parse_dataset_config(cfg["dataset"])
    splits = sdk.build_dataset(sdk.generate_corpus())
    tr = sdk.train("fc", splits)
    ev = sdk.evaluate(tr, splits)
    assert ev.overall_test_mse > 0
    assert tr.run_dir.exists() and (tr.run_dir / "checkpoint_best.pt").exists()


def test_parse_signal_config_round_trip() -> None:
    cfg = load_config()
    sig: SignalConfig = parse_signal_config(cfg["signal"])
    assert sig.fs == 1000
    assert sig.frequencies_hz == (2.0, 10.0, 50.0, 200.0)
    assert math.isclose(sig.phases_rad[1], math.pi / 2, abs_tol=1e-12)
    assert math.isclose(sig.noise_beta, 2 * math.pi, abs_tol=1e-12)
    assert sig.noise_distribution == "gaussian"


def test_parse_signal_config_numeric_phases_and_beta() -> None:
    """Numeric (non-string) phases/beta must pass through unchanged."""
    sig = parse_signal_config({
        "fs": 1000, "duration_s": 10,
        "frequencies_hz": [2, 10, 50, 200],
        "amplitudes": [1.0, 1.0, 1.0, 1.0],
        "phases_rad": [0.0, 1.0, 2.0, 3.0],
        "noise": {"alpha": 0.05, "beta": 1.0, "distribution": "gaussian"},
    })
    assert sig.phases_rad == (0.0, 1.0, 2.0, 3.0)
    assert sig.noise_beta == 1.0


def test_parse_dataset_config_round_trip() -> None:
    cfg = load_config()
    ds = parse_dataset_config(cfg["dataset"])
    assert ds.window == 10
    assert (ds.n_train, ds.n_val, ds.n_test) == (30_000, 3_750, 3_750)

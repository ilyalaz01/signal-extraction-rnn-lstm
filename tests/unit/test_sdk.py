"""Tests for the SDK layer — init, generate_corpus, build_dataset.

The unimplemented methods (train, evaluate, run_experiment) raise
NotImplementedError("M4"); this is asserted to pin the surface.
"""

import math

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


def test_sdk_init_default() -> None:
    sdk = SDK()
    assert sdk.seed == 1337
    assert sdk.device.type in ("cpu", "cuda")


def test_sdk_init_seed_override() -> None:
    sdk = SDK(seed=99)
    assert sdk.seed == 99


def test_sdk_init_device_override() -> None:
    sdk = SDK(device="cpu")
    assert sdk.device == torch.device("cpu")


def test_sdk_generate_corpus_default_shape() -> None:
    sdk = SDK()
    c = sdk.generate_corpus()
    assert c.clean.shape == (4, 10_000)
    assert c.noisy.shape == (4, 10_000)
    assert c.config.fs == 1000


def test_sdk_build_dataset_split_lengths() -> None:
    sdk = SDK()
    splits = sdk.build_dataset(sdk.generate_corpus())
    assert (len(splits.train), len(splits.val), len(splits.test)) == (30_000, 3_750, 3_750)


def test_sdk_determinism_same_seed() -> None:
    a, b = SDK(seed=42), SDK(seed=42)
    np.testing.assert_array_equal(a.generate_corpus().noisy, b.generate_corpus().noisy)


def test_sdk_different_seeds_yield_different_corpora() -> None:
    a, b = SDK(seed=42), SDK(seed=43)
    assert not np.array_equal(a.generate_corpus().noisy, b.generate_corpus().noisy)


def test_sdk_unimplemented_methods_raise() -> None:
    sdk = SDK()
    with pytest.raises(NotImplementedError):
        sdk.train("fc", None)
    with pytest.raises(NotImplementedError):
        sdk.evaluate(None, None)
    with pytest.raises(NotImplementedError):
        sdk.run_experiment(None)


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

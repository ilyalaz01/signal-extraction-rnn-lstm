"""Tests for services.evaluation — T-EV-01..06."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from signal_extraction_rnn_lstm.services.dataset import (
    DatasetConfig,
    SplitDatasets,
    build_split_datasets,
)
from signal_extraction_rnn_lstm.services.evaluation import EvalResult, evaluate
from signal_extraction_rnn_lstm.services.models.base import SignalExtractor
from signal_extraction_rnn_lstm.services.models.fc import FCConfig, FCExtractor
from signal_extraction_rnn_lstm.services.signal_gen import (
    Corpus,
    SignalConfig,
    generate_corpus,
)


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return generate_corpus(SignalConfig(
        fs=1000, duration_s=10, frequencies_hz=(2.0, 10.0, 50.0, 200.0),
        amplitudes=(1.0, 1.0, 1.0, 1.0),
        phases_rad=(0.0, math.pi / 2, math.pi, 3 * math.pi / 2),
        noise_alpha=0.05, noise_beta=2 * math.pi, noise_distribution="gaussian",
    ), seed=42)


@pytest.fixture
def splits(corpus: Corpus) -> SplitDatasets:
    return build_split_datasets(
        corpus, DatasetConfig(window=10, n_train=200, n_val=50, n_test=200), sampling_seed=7,
    )


@pytest.fixture
def untrained_model() -> FCExtractor:
    torch.manual_seed(0)
    return FCExtractor(FCConfig())


def test_t_ev_01_output_types(untrained_model: FCExtractor, splits: SplitDatasets,
                              tmp_path: Path) -> None:
    res = evaluate(untrained_model, splits, tmp_path)
    assert isinstance(res, EvalResult)
    assert isinstance(res.overall_test_mse, float)
    assert isinstance(res.per_freq_mse, dict)
    assert set(res.per_freq_mse.keys()) == {0, 1, 2, 3}
    for v in res.per_freq_mse.values():
        assert isinstance(v, float)


def test_t_ev_02_untrained_nontrivial_mse(untrained_model: FCExtractor,
                                           splits: SplitDatasets, tmp_path: Path) -> None:
    res = evaluate(untrained_model, splits, tmp_path)
    assert res.overall_test_mse > 0 and math.isfinite(res.overall_test_mse)


def test_t_ev_03_per_frequency_correctness(untrained_model: FCExtractor,
                                            splits: SplitDatasets, tmp_path: Path) -> None:
    """Re-compute per-frequency MSE the long way and compare."""
    res = evaluate(untrained_model, splits, tmp_path)
    # Manual loop: gather predictions/targets and group by selector argmax.
    untrained_model.eval()
    preds, targets, ks = [], [], []
    with torch.no_grad():
        for batch in DataLoader(splits.test, batch_size=256, shuffle=False):
            preds.append(untrained_model(batch.selector, batch.w_noisy))
            targets.append(batch.w_clean)
            ks.append(batch.selector.argmax(dim=-1))
    pred, target, k_all = torch.cat(preds), torch.cat(targets), torch.cat(ks)
    for k in range(4):
        mask = k_all == k
        manual = float(nn.functional.mse_loss(pred[mask], target[mask]))
        assert abs(res.per_freq_mse[k] - manual) < 1e-6


def test_t_ev_04_per_freq_hz_property(untrained_model: FCExtractor,
                                       splits: SplitDatasets, tmp_path: Path) -> None:
    res = evaluate(untrained_model, splits, tmp_path)
    assert res.per_freq_hz[2.0] == res.per_freq_mse[0]
    assert res.per_freq_hz[200.0] == res.per_freq_mse[3]
    assert set(res.per_freq_hz.keys()) == {2.0, 10.0, 50.0, 200.0}


def test_t_ev_05_results_json_schema(untrained_model: FCExtractor, splits: SplitDatasets,
                                      tmp_path: Path) -> None:
    evaluate(untrained_model, splits, tmp_path)
    payload = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert set(payload.keys()) == {"version", "spec", "training", "evaluation"}
    assert payload["version"] == "1.00"
    assert set(payload["evaluation"]["per_freq_mse"].keys()) == {"0", "1", "2", "3"}


def test_t_ev_06_perfect_model_zero_mse(corpus: Corpus, tmp_path: Path) -> None:
    """Build a corpus whose clean channels are all zero; a zero-predictor reaches MSE = 0."""
    zero_clean = np.zeros_like(corpus.clean)
    zero_corpus = replace(corpus, clean=zero_clean.astype(np.float32))
    zsplits = build_split_datasets(
        zero_corpus, DatasetConfig(window=10, n_train=4, n_val=4, n_test=40), sampling_seed=0,
    )

    class ZeroPredictor(SignalExtractor):
        def forward(self, sel: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
            return torch.zeros(sel.shape[0], 10)

    res = evaluate(ZeroPredictor(), zsplits, tmp_path)
    assert res.overall_test_mse == 0.0
    for k in range(4):
        assert res.per_freq_mse[k] == 0.0


def test_evaluate_restores_training_mode(untrained_model: FCExtractor,
                                          splits: SplitDatasets, tmp_path: Path) -> None:
    untrained_model.train(True)
    evaluate(untrained_model, splits, tmp_path)
    assert untrained_model.training is True
    untrained_model.train(False)
    evaluate(untrained_model, splits, tmp_path)
    assert untrained_model.training is False


def test_evaluate_handles_empty_frequency_group(corpus: Corpus, tmp_path: Path) -> None:
    """If a test split misses one channel, that group's MSE is NaN (PRD § 13)."""
    splits = build_split_datasets(
        corpus, DatasetConfig(window=10, n_train=4, n_val=4, n_test=4), sampling_seed=0,
    )
    # Force the test index_table to only contain selector k=0
    splits.test.index_table[:, 1] = 0
    torch.manual_seed(0)
    res = evaluate(FCExtractor(FCConfig()), splits, tmp_path)
    for k in (1, 2, 3):
        assert math.isnan(res.per_freq_mse[k])

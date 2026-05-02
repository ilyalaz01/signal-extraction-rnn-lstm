"""RED-phase tests for services.dataset — T-DS-04..18.

T-DS-01..03 retired in PRD v1.01 (compute_split_ranges removed).
T-DS-11' replaces v1.00 T-DS-11 (KS equivalence under shared pool).
"""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from signal_extraction_rnn_lstm.services.dataset import (
    DatasetConfig,
    SplitDatasets,
    WindowDataset,
    WindowExample,
    build_split_datasets,
)
from signal_extraction_rnn_lstm.services.signal_gen import (
    Corpus,
    SignalConfig,
    generate_corpus,
)


def _ks_two_sample_p(a: np.ndarray, b: np.ndarray) -> float:
    """Asymptotic two-sample KS p-value (Smirnov approximation)."""
    a, b = np.sort(a), np.sort(b)
    all_v = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_v, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_v, side="right") / len(b)
    d = float(np.abs(cdf_a - cdf_b).max())
    en = math.sqrt(len(a) * len(b) / (len(a) + len(b)))
    return 2.0 * math.exp(-2.0 * (en * d) ** 2)


def _ks_uniform_p(x: np.ndarray, low: float, high: float) -> float:
    """Asymptotic one-sample KS p-value vs Uniform[low, high]."""
    xs = np.sort(x)
    cdf_emp = np.arange(1, len(xs) + 1) / len(xs)
    cdf_ref = (xs - low) / (high - low)
    d = float(np.max(np.abs(cdf_emp - cdf_ref)))
    return 2.0 * math.exp(-2.0 * len(xs) * d * d)


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return generate_corpus(SignalConfig(
        fs=1000, duration_s=10, frequencies_hz=(2.0, 10.0, 50.0, 200.0),
        amplitudes=(1.0, 1.0, 1.0, 1.0),
        phases_rad=(0.0, math.pi / 2, math.pi, 3 * math.pi / 2),
        noise_alpha=0.05, noise_beta=2 * math.pi, noise_distribution="gaussian",
    ), seed=42)


@pytest.fixture
def cfg() -> DatasetConfig:
    return DatasetConfig(window=10, n_train=30_000, n_val=3_750, n_test=3_750)


@pytest.fixture
def splits(corpus: Corpus, cfg: DatasetConfig) -> SplitDatasets:
    return build_split_datasets(corpus, cfg, sampling_seed=7)


def test_t_ds_04_window_locked() -> None:
    DatasetConfig(window=10, n_train=10, n_val=10, n_test=10)
    for w in (8, 9, 11, 16):
        with pytest.raises(ValueError):
            DatasetConfig(window=w, n_train=10, n_val=10, n_test=10)


def test_t_ds_05_lengths(splits: SplitDatasets, cfg: DatasetConfig) -> None:
    assert (len(splits.train), len(splits.val), len(splits.test)) == (
        cfg.n_train, cfg.n_val, cfg.n_test,
    )


def test_t_ds_06_determinism(corpus: Corpus, cfg: DatasetConfig) -> None:
    a = build_split_datasets(corpus, cfg, sampling_seed=7)
    b = build_split_datasets(corpus, cfg, sampling_seed=7)
    np.testing.assert_array_equal(a.train.index_table, b.train.index_table)
    torch.testing.assert_close(a.train[0].w_clean, b.train[0].w_clean)


def test_t_ds_07_index_table_dtype(splits: SplitDatasets) -> None:
    table = splits.train.index_table
    assert table.dtype == np.int64
    assert table.shape == (len(splits.train), 2)


def test_t_ds_08_getitem_shapes(splits: SplitDatasets) -> None:
    e = splits.train[0]
    assert (e.selector.shape, e.w_noisy.shape, e.w_clean.shape) == ((4,), (10,), (10,))
    assert e.selector.dtype == e.w_noisy.dtype == e.w_clean.dtype == torch.float32


def test_t_ds_09_one_hot_valid(splits: SplitDatasets) -> None:
    for i in (0, 100, len(splits.train) - 1):
        s = splits.train[i].selector
        assert float(s.sum()) == 1.0 and int((s == 1.0).sum()) == 1


def test_t_ds_10_selector_target_consistency(corpus: Corpus, splits: SplitDatasets) -> None:
    for i in (0, 1234, 12_345):
        e = splits.train[i]
        m = splits.train.meta(i)
        assert int(e.selector.argmax()) == m["k"]
        t0, k = m["t_0"], m["k"]
        np.testing.assert_array_equal(e.w_clean.numpy(), corpus.clean[k, t0:t0 + 10])
        np.testing.assert_array_equal(e.w_noisy.numpy(), corpus.noisy_sum[t0:t0 + 10])


def test_t_ds_11_split_distributions_equivalent(splits: SplitDatasets) -> None:
    p = _ks_two_sample_p(
        splits.train.index_table[:, 0].astype(float),
        splits.test.index_table[:, 0].astype(float),
    )
    assert p > 0.001


def test_t_ds_12_class_balance(splits: SplitDatasets) -> None:
    counts = np.bincount(splits.train.index_table[:, 1], minlength=4)
    sigma = math.sqrt(30_000 * 0.25 * 0.75)
    assert all(abs(c - 7500) <= 4 * sigma for c in counts)


def test_t_ds_13_t0_uniform(splits: SplitDatasets) -> None:
    assert _ks_uniform_p(splits.train.index_table[:, 0].astype(float), 0.0, 9990.0) > 0.001


def test_t_ds_14_dataloader_integration(splits: SplitDatasets) -> None:
    batch = next(iter(DataLoader(splits.train, batch_size=256)))
    assert (batch.selector.shape, batch.w_noisy.shape, batch.w_clean.shape) == (
        (256, 4), (256, 10), (256, 10),
    )


def test_t_ds_15_memory_sharing(corpus: Corpus, splits: SplitDatasets) -> None:
    for ds in (splits.train, splits.val, splits.test):
        assert isinstance(ds, WindowDataset)
        assert ds.corpus.clean is corpus.clean and ds.corpus.noisy_sum is corpus.noisy_sum


def test_t_ds_16_seed_independence(corpus: Corpus, cfg: DatasetConfig) -> None:
    a = build_split_datasets(corpus, cfg, sampling_seed=1)
    b = build_split_datasets(corpus, cfg, sampling_seed=2)
    assert not np.array_equal(a.train.index_table, b.train.index_table)


def test_t_ds_17_train_order_preserved(corpus: Corpus) -> None:
    cfg_a = DatasetConfig(window=10, n_train=1_000, n_val=100, n_test=100)
    cfg_b = DatasetConfig(window=10, n_train=1_000, n_val=100, n_test=200)
    a = build_split_datasets(corpus, cfg_a, sampling_seed=11)
    b = build_split_datasets(corpus, cfg_b, sampling_seed=11)
    np.testing.assert_array_equal(a.train.index_table, b.train.index_table)


def test_t_ds_18_meta_idempotent(splits: SplitDatasets) -> None:
    for i in (0, 100, len(splits.train) - 1):
        m = splits.train.meta(i)
        assert m == splits.train.meta(i)
        assert set(m) == {"t_0", "k", "split_name"} and m["split_name"] == "train"
        assert 0 <= m["t_0"] <= 9_990 and m["k"] in (0, 1, 2, 3)
    assert splits.val.meta(0)["split_name"] == "val"
    assert splits.test.meta(0)["split_name"] == "test"


def test_zero_or_negative_split_counts_raise() -> None:
    with pytest.raises(ValueError):
        DatasetConfig(window=10, n_train=0, n_val=0, n_test=0)
    with pytest.raises(ValueError):
        DatasetConfig(window=10, n_train=-1, n_val=10, n_test=10)


def test_seed_none_raises(corpus: Corpus, cfg: DatasetConfig) -> None:
    with pytest.raises(TypeError):
        build_split_datasets(corpus, cfg, sampling_seed=None)  # type: ignore[arg-type]


def test_corpus_too_short_raises(cfg: DatasetConfig) -> None:
    short = generate_corpus(SignalConfig(
        fs=1, duration_s=1, frequencies_hz=(0.1, 0.2, 0.3, 0.4),
        amplitudes=(1.0,) * 4, phases_rad=(0.0,) * 4,
        noise_alpha=0.0, noise_beta=0.0, noise_distribution="gaussian",
    ), seed=0)
    with pytest.raises(ValueError):
        build_split_datasets(short, cfg, sampling_seed=0)


def test_corpus_wrong_channel_count_raises(corpus: Corpus, cfg: DatasetConfig) -> None:
    bad = replace(corpus, clean=corpus.clean[:3])  # 3 channels instead of 4
    with pytest.raises(ValueError):
        build_split_datasets(bad, cfg, sampling_seed=0)


def test_window_example_is_namedtuple() -> None:
    """Pin the impl choice — NamedTuple, so default_collate works (PROMPTS § 6)."""
    assert issubclass(WindowExample, tuple)
    assert WindowExample._fields == ("selector", "w_noisy", "w_clean")

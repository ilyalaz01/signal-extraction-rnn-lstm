"""Dataset construction service.

Window sampling, train/val/test splitting, and torch ``Dataset`` wrappers
per ``PRD_dataset_construction.md`` v1.01 and ADR-016 / ADR-017.

Public surface:
    DatasetConfig                 (frozen dataclass)
    WindowExample                 (NamedTuple — see PROMPTS session 6)
    WindowDataset                 (torch.utils.data.Dataset) with .meta(i)
    SplitDatasets                 (frozen dataclass)
    build_split_datasets(corpus, config, sampling_seed) → SplitDatasets
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset

from signal_extraction_rnn_lstm.constants import N_SINUSOIDS, WINDOW_SIZE
from signal_extraction_rnn_lstm.services.signal_gen import Corpus

_SPLIT_NAMES = ("train", "val", "test")


class WindowExample(NamedTuple):
    """One window-sampled example.

    Input:  one ``__getitem__`` call on a ``WindowDataset``.
    Output: this triple (selector, w_noisy, w_clean).
    Setup:  NamedTuple — chosen over ``@dataclass(frozen=True)`` to interop
            with ``torch.utils.data.default_collate``, which collates
            NamedTuples field-wise but does not handle dataclasses across
            our supported torch range.  Field-level interface is identical
            to the ``@dataclass`` form sketched in PRD § 3.3.
    """

    selector: torch.Tensor    # (4,)  float32, exactly one entry == 1.0
    w_noisy: torch.Tensor     # (10,) float32
    w_clean: torch.Tensor     # (10,) float32


@dataclass(frozen=True)
class DatasetConfig:
    """Parsed and validated form of ``config['dataset']``.

    Input:  individual fields from the JSON config.
    Output: frozen, validated config; ``ValueError`` on invalid input.
    Setup:  ``__post_init__`` enforces window=10 (HOMEWORK_BRIEF § 4.2)
            and at-least-one-non-empty-split.
    """

    window: int
    n_train: int
    n_val: int
    n_test: int

    def __post_init__(self) -> None:
        if self.window != WINDOW_SIZE:
            raise ValueError(f"window must be {WINDOW_SIZE}, got {self.window}")
        for name, n in (("n_train", self.n_train), ("n_val", self.n_val),
                        ("n_test", self.n_test)):
            if n < 0:
                raise ValueError(f"{name} must be >= 0, got {n}")
        if self.n_train + self.n_val + self.n_test == 0:
            raise ValueError("at least one split must be non-empty")


class WindowDataset(Dataset):
    """A split-specific dataset of (selector, w_noisy, w_clean) triples.

    Input:  index ``i ∈ [0, len-1)``.
    Output: ``WindowExample`` at position ``i`` (deterministic).
    Setup:  carries a reference to the corpus (no copy), the (n, 2) int64
            index table, and the ``split_name`` used by ``meta()``.
    """

    def __init__(self, corpus: Corpus, index_table: np.ndarray, split_name: str) -> None:
        self.corpus = corpus
        self.index_table = index_table
        self.split_name = split_name

    def __len__(self) -> int:
        return self.index_table.shape[0]

    def __getitem__(self, i: int) -> WindowExample:
        t0 = int(self.index_table[i, 0])
        k = int(self.index_table[i, 1])
        selector = torch.zeros(N_SINUSOIDS, dtype=torch.float32)
        selector[k] = 1.0
        w_noisy = torch.from_numpy(self.corpus.noisy_sum[t0:t0 + WINDOW_SIZE])
        w_clean = torch.from_numpy(self.corpus.clean[k, t0:t0 + WINDOW_SIZE])
        return WindowExample(selector, w_noisy, w_clean)

    def meta(self, i: int) -> dict:
        """Return ``{'t_0', 'k', 'split_name'}`` for the example at ``i``.

        Pure function of ``(i, index_table, split_name)`` — idempotent across
        repeated calls; does not interact with ``__getitem__``.
        """
        return {
            "t_0": int(self.index_table[i, 0]),
            "k": int(self.index_table[i, 1]),
            "split_name": self.split_name,
        }


@dataclass(frozen=True)
class SplitDatasets:
    """The three populated WindowDatasets plus provenance.

    Input:  produced by ``build_split_datasets(corpus, config, seed)``.
    Output: this object.
    Setup:  the three datasets share the corpus reference (T-DS-15).
    """

    train: WindowDataset
    val: WindowDataset
    test: WindowDataset
    config: DatasetConfig
    corpus_seed: int
    sampling_seed: int


def _sample_index_table(rng: np.random.Generator, n: int, n_pool: int) -> np.ndarray:
    t0 = rng.integers(0, n_pool, size=n, dtype=np.int64)
    k = rng.integers(0, N_SINUSOIDS, size=n, dtype=np.int64)
    return np.stack([t0, k], axis=1)


def build_split_datasets(corpus: Corpus, config: DatasetConfig,
                         sampling_seed: int) -> SplitDatasets:
    """Construct three ``WindowDataset``s sharing one ``Corpus``.

    Input:  ``Corpus``, ``DatasetConfig``, integer seed (None forbidden).
    Output: ``SplitDatasets`` — train/val/test populated, plus provenance.
    Setup:  ``np.random.SeedSequence(seed).spawn(3)`` yields three child
            seeds (train, val, test in order).  Sampling is iid uniform on
            ``[0, N - W] × {0, 1, 2, 3}`` with replacement, per ADR-016.
            Spawning order means changing ``n_test`` does not perturb the
            train table (verified by T-DS-17).
    """
    if (sampling_seed is None or not isinstance(sampling_seed, int)
            or isinstance(sampling_seed, bool)):
        raise TypeError(
            f"sampling_seed must be a non-None int, got {type(sampling_seed).__name__}"
        )
    if corpus.clean.shape[0] != N_SINUSOIDS:
        raise ValueError(
            f"corpus.clean must have {N_SINUSOIDS} channels, got {corpus.clean.shape[0]}"
        )
    n_pool = corpus.n_samples - config.window + 1
    if n_pool <= 0:
        raise ValueError(
            f"corpus.n_samples ({corpus.n_samples}) < window ({config.window})"
        )
    train_seed, val_seed, test_seed = np.random.SeedSequence(sampling_seed).spawn(3)
    sizes = (config.n_train, config.n_val, config.n_test)
    seeds = (train_seed, val_seed, test_seed)
    tables = [
        _sample_index_table(np.random.default_rng(s), n, n_pool)
        for s, n in zip(seeds, sizes, strict=True)
    ]
    datasets = [
        WindowDataset(corpus, tables[i], _SPLIT_NAMES[i]) for i in range(3)
    ]
    return SplitDatasets(
        train=datasets[0], val=datasets[1], test=datasets[2],
        config=config, corpus_seed=corpus.seed, sampling_seed=sampling_seed,
    )

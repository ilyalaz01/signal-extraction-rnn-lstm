"""Integration smoke — SDK end-to-end (AC-DS-7).

Exercises the canonical user path:
    SDK() → generate_corpus() → build_dataset() → DataLoader → first batch.
Pins that the SDK contract works on the real config without mocks.
"""

import torch
from torch.utils.data import DataLoader

from signal_extraction_rnn_lstm.sdk import SDK


def test_sdk_smoke_end_to_end() -> None:
    sdk = SDK(seed=0, device="cpu")
    corpus = sdk.generate_corpus()
    splits = sdk.build_dataset(corpus)

    assert len(splits.train) > 0
    assert len(splits.val) > 0
    assert len(splits.test) > 0

    e = splits.train[0]
    assert e.selector.shape == (4,)
    assert e.w_noisy.shape == (10,)
    assert e.w_clean.shape == (10,)

    batch = next(iter(DataLoader(splits.train, batch_size=32)))
    assert batch.selector.shape == (32, 4)
    assert batch.w_noisy.shape == (32, 10)
    assert batch.w_clean.dtype == torch.float32


def test_sdk_smoke_seed_then_seed_reproducible() -> None:
    """One knob — same seed → same dataset across full SDK construction."""
    s1 = SDK(seed=7, device="cpu")
    s2 = SDK(seed=7, device="cpu")
    a = s1.build_dataset(s1.generate_corpus())
    b = s2.build_dataset(s2.generate_corpus())
    e_a, e_b = a.train[0], b.train[0]
    torch.testing.assert_close(e_a.selector, e_b.selector)
    torch.testing.assert_close(e_a.w_noisy, e_b.w_noisy)
    torch.testing.assert_close(e_a.w_clean, e_b.w_clean)

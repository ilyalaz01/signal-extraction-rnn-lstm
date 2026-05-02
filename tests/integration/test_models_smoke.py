"""AC-MD-6: SDK → real WindowDataset batch → each model → finite loss.

Exercises the full data path with each architecture for one mini-batch,
matching the contract the training loop will rely on in M4.
"""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from signal_extraction_rnn_lstm.sdk import SDK
from signal_extraction_rnn_lstm.services.models import build, parse_model_config
from signal_extraction_rnn_lstm.shared.config import load_config


@pytest.mark.parametrize("kind", ["fc", "rnn", "lstm"])
def test_models_smoke_each_kind_with_real_data(kind: str) -> None:
    sdk = SDK(seed=0, device="cpu")
    splits = sdk.build_dataset(sdk.generate_corpus())
    mcfg = parse_model_config(load_config()["model"])

    batch = next(iter(DataLoader(splits.train, batch_size=32)))
    torch.manual_seed(0)
    model = build(kind, mcfg)
    out = model(batch.selector, batch.w_noisy)

    assert out.shape == (32, 10)
    assert out.dtype == torch.float32
    loss = nn.functional.mse_loss(out, batch.w_clean)
    assert torch.isfinite(loss).item(), f"{kind} produced non-finite loss"

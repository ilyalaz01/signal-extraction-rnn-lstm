"""Tests for shared.config.load_config (parse_angle is covered in test_signal_gen)."""

import json
from pathlib import Path

import pytest

from signal_extraction_rnn_lstm.shared.config import (
    ConfigVersionMismatchError,
    apply_overrides,
    load_config,
)


def test_load_config_default_path() -> None:
    cfg = load_config()
    assert cfg["version"] == "1.00"
    assert "signal" in cfg
    assert cfg["signal"]["fs"] == 1000


def test_load_config_explicit_path(tmp_path: Path) -> None:
    f = tmp_path / "ok.json"
    f.write_text(json.dumps({"version": "1.00", "signal": {}}))
    cfg = load_config(f)
    assert cfg["version"] == "1.00"


def test_load_config_version_mismatch(tmp_path: Path) -> None:
    f = tmp_path / "bad.json"
    f.write_text(json.dumps({"version": "9.99"}))
    with pytest.raises(ConfigVersionMismatchError):
        load_config(f)


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.json")


def test_apply_overrides_deep_merge_leaf() -> None:
    cfg = {"signal": {"noise": {"alpha": 0.05, "beta": "2*pi"}}, "training": {"lr": 1e-3}}
    out = apply_overrides(cfg, {"signal.noise.alpha": 0.20, "training.lr": 5e-4})
    assert out["signal"]["noise"]["alpha"] == 0.20
    assert out["training"]["lr"] == 5e-4
    # original untouched
    assert cfg["signal"]["noise"]["alpha"] == 0.05


def test_apply_overrides_unknown_path_raises() -> None:
    cfg = {"signal": {"noise": {"alpha": 0.05}}}
    with pytest.raises(KeyError):
        apply_overrides(cfg, {"signal.noise.gamma": 1.0})
    with pytest.raises(KeyError):
        apply_overrides(cfg, {"unknown.path": 1.0})


def test_apply_overrides_empty_dict_is_noop() -> None:
    cfg = {"a": 1, "b": {"c": 2}}
    assert apply_overrides(cfg, {}) == cfg

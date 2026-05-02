"""Thin CLI wrapper: run a full experiment (train + evaluate) for one model kind.

Usage:
    uv run python scripts/run_experiment.py --kind fc [--seed N] [--override key=value]...

``--override`` may be passed multiple times.  Each override is a dotted-path
into ``config/setup.json`` (e.g. ``signal.noise.alpha=0.20``).
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from signal_extraction_rnn_lstm.sdk import SDK
from signal_extraction_rnn_lstm.sdk.sdk import ExperimentSpec


def _parse_override(s: str) -> tuple[str, Any]:
    if "=" not in s:
        raise ValueError(f"override must be 'key=value', got {s!r}")
    key, raw = s.split("=", 1)
    try:
        value = json.loads(raw)  # 1, 1.0, true, "string", etc.
    except json.JSONDecodeError:
        value = raw
    return key, value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["fc", "rnn", "lstm"], required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--override", action="append", default=[],
                        help="dotted-path=JSON-value; may be repeated")
    args = parser.parse_args()

    overrides = dict(_parse_override(o) for o in args.override)
    spec = ExperimentSpec(model_kind=args.kind, seed=args.seed, overrides=overrides)
    sdk = SDK(seed=args.seed)
    result = sdk.run_experiment(spec)
    print(f"experiment {args.kind} (seed={sdk.seed if args.seed is None else args.seed}): "
          f"overall test MSE {result.eval_result.overall_test_mse:.6f}")
    print(f"run dir: {result.train_result.run_dir}")


if __name__ == "__main__":
    main()

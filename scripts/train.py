"""Thin CLI wrapper: train one model kind on the default config corpus.

Usage:
    uv run python scripts/train.py [--kind fc|rnn|lstm] [--seed N]

All runs land in ``results/<utc_timestamp>__<kind>__<seed>/`` per ADR-014.
"""

from __future__ import annotations

import argparse

from signal_extraction_rnn_lstm.sdk import SDK


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["fc", "rnn", "lstm"], default="fc")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    sdk = SDK(seed=args.seed)
    splits = sdk.build_dataset(sdk.generate_corpus())
    result = sdk.train(args.kind, splits)
    print(f"trained {args.kind} (seed={sdk.seed}); "
          f"best epoch {result.best_epoch}, val MSE {result.best_val_mse:.6f}")
    print(f"run dir: {result.run_dir}")


if __name__ == "__main__":
    main()

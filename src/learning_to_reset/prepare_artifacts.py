"""CLI entrypoint for exporting prepared datasets."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from learning_to_reset.dataset_prep import export_prepared_datasets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare reset-aware SFT and Countdown JSONL artifacts."
    )
    parser.add_argument("--traces", required=True, help="Path to the trace JSON/JSONL file.")
    parser.add_argument("--countdown", required=True, help="Path to the Countdown JSON/JSONL file.")
    parser.add_argument("--output-dir", required=True, help="Directory for prepared JSONL outputs.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument(
        "--allow-clean-eval",
        action="store_true",
        help="Allow reset instructions in Countdown evaluation prompts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    manifest = export_prepared_datasets(
        trace_path=args.traces,
        countdown_path=args.countdown,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        allow_clean_eval=args.allow_clean_eval,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

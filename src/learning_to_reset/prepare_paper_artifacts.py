"""CLI for preparing paper-aligned artifacts from real source files."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from learning_to_reset.paper_dataset_prep import export_paper_prepared_datasets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare SFT and Countdown artifacts from separate source train/eval files."
    )
    parser.add_argument("--traces", required=True, help="Path to the trace JSON/JSONL file.")
    parser.add_argument("--countdown-train", required=True, help="Path to the Countdown train JSON/JSONL file.")
    parser.add_argument("--countdown-eval", required=True, help="Path to the Countdown eval JSON/JSONL file.")
    parser.add_argument("--output-dir", required=True, help="Directory for prepared JSONL outputs.")
    parser.add_argument("--sft-val-ratio", type=float, default=0.1, help="Validation ratio for SFT traces.")
    parser.add_argument(
        "--countdown-val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio carved out of the Countdown train file.",
    )
    parser.add_argument(
        "--disallow-clean",
        action="store_true",
        help="Do not include clean instructions in prepared Countdown prompts.",
    )
    parser.add_argument(
        "--include-recovery-examples",
        action="store_true",
        help="Append retry-stage recovery examples for traces that provide explicit recovery responses.",
    )
    parser.add_argument(
        "--recovery-repeat",
        type=int,
        default=1,
        help="How many times to repeat each retry-stage recovery example when recovery augmentation is enabled.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = export_paper_prepared_datasets(
        trace_path=args.traces,
        countdown_train_path=args.countdown_train,
        countdown_eval_path=args.countdown_eval,
        output_dir=args.output_dir,
        sft_val_ratio=args.sft_val_ratio,
        countdown_val_ratio=args.countdown_val_ratio,
        allow_clean=not args.disallow_clean,
        include_recovery_examples=args.include_recovery_examples,
        recovery_repeat=args.recovery_repeat,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

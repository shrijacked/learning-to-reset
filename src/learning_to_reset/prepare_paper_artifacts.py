"""CLI for preparing paper-aligned artifacts from real source files."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Mapping, Optional, Sequence

from learning_to_reset.paper_dataset_prep import export_paper_prepared_datasets


SCALE_PRESETS: Dict[str, Dict[str, Any]] = {
    "pilot": {
        "sft_val_ratio": 0.1,
        "countdown_val_ratio": 0.1,
        "hard_mine_ratio": 0.2,
        "recovery_repeat": 1,
        "include_recovery_examples": False,
        "require_recovery_target_correct": False,
    },
    "paper": {
        "sft_val_ratio": 0.05,
        "countdown_val_ratio": 0.05,
        "hard_mine_ratio": 0.2,
        "recovery_repeat": 4,
        "include_recovery_examples": True,
        "require_recovery_target_correct": True,
    },
}


def apply_scale_preset(
    *,
    scale: str,
    overrides: Mapping[str, Any],
) -> Dict[str, Any]:
    """Fill in any missing scale knob with its preset value.

    Explicit CLI overrides win: if ``overrides[key]`` is not ``None`` it is
    kept verbatim. If ``overrides[key]`` is ``None`` the preset value is
    substituted. Unknown keys in ``overrides`` are passed through unchanged.
    """

    if scale not in SCALE_PRESETS:
        raise ValueError(
            f"Unknown scale preset {scale!r}. Choose from {sorted(SCALE_PRESETS)!r}."
        )
    preset = SCALE_PRESETS[scale]
    configured: Dict[str, Any] = dict(overrides)
    for key, preset_value in preset.items():
        if configured.get(key) is None:
            configured[key] = preset_value
    return configured


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare SFT and Countdown artifacts from separate source train/eval files."
    )
    parser.add_argument("--traces", required=True, help="Path to the trace JSON/JSONL file.")
    parser.add_argument("--countdown-train", required=True, help="Path to the Countdown train JSON/JSONL file.")
    parser.add_argument("--countdown-eval", required=True, help="Path to the Countdown eval JSON/JSONL file.")
    parser.add_argument("--output-dir", required=True, help="Directory for prepared JSONL outputs.")
    parser.add_argument(
        "--scale",
        choices=sorted(SCALE_PRESETS),
        default="pilot",
        help=(
            "Preparation recipe preset. 'pilot' (default) is the local CPU/MPS "
            "configuration; 'paper' matches Section 4.1 of main.pdf "
            "(recovery_repeat=4, target-correct recovery filter on)."
        ),
    )
    parser.add_argument("--sft-val-ratio", type=float, default=None, help="Validation ratio for SFT traces.")
    parser.add_argument(
        "--countdown-val-ratio",
        type=float,
        default=None,
        help="Validation ratio carved out of the Countdown train file.",
    )
    parser.add_argument(
        "--hard-mine-ratio",
        type=float,
        default=None,
        help=(
            "Fraction of the hard Countdown training slice reserved for failure mining "
            "instead of hard-train."
        ),
    )
    parser.add_argument(
        "--disallow-clean",
        action="store_true",
        help="Do not include clean instructions in prepared Countdown prompts.",
    )
    parser.add_argument(
        "--include-recovery-examples",
        dest="include_recovery_examples",
        action="store_true",
        default=None,
        help="Append retry-stage recovery examples for traces that provide explicit recovery responses.",
    )
    parser.add_argument(
        "--no-include-recovery-examples",
        dest="include_recovery_examples",
        action="store_false",
        help="Override the scale preset and skip retry-stage recovery examples.",
    )
    parser.add_argument(
        "--recovery-repeat",
        type=int,
        default=None,
        help="How many times to repeat each retry-stage recovery example when recovery augmentation is enabled.",
    )
    parser.add_argument(
        "--require-recovery-target-correct",
        dest="require_recovery_target_correct",
        action="store_true",
        default=None,
        help="Only append retry-stage recovery examples whose final answer verifies against Countdown metadata.",
    )
    parser.add_argument(
        "--no-require-recovery-target-correct",
        dest="require_recovery_target_correct",
        action="store_false",
        help="Override the scale preset and skip the target-correct filter.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    configured = apply_scale_preset(
        scale=args.scale,
        overrides={
            "sft_val_ratio": args.sft_val_ratio,
            "countdown_val_ratio": args.countdown_val_ratio,
            "hard_mine_ratio": args.hard_mine_ratio,
            "recovery_repeat": args.recovery_repeat,
            "include_recovery_examples": args.include_recovery_examples,
            "require_recovery_target_correct": args.require_recovery_target_correct,
        },
    )
    manifest = export_paper_prepared_datasets(
        trace_path=args.traces,
        countdown_train_path=args.countdown_train,
        countdown_eval_path=args.countdown_eval,
        output_dir=args.output_dir,
        sft_val_ratio=configured["sft_val_ratio"],
        countdown_val_ratio=configured["countdown_val_ratio"],
        hard_mine_ratio=configured["hard_mine_ratio"],
        allow_clean=not args.disallow_clean,
        include_recovery_examples=bool(configured["include_recovery_examples"]),
        recovery_repeat=int(configured["recovery_repeat"]),
        require_recovery_target_correct=bool(
            configured["require_recovery_target_correct"]
        ),
    )
    manifest["scale"] = args.scale
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Prepare paper-aligned artifacts from separate source train/eval files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from learning_to_reset.countdown_slices import filter_hard_countdown_samples
from learning_to_reset.data import load_countdown_samples, load_trace_records
from learning_to_reset.dataset_prep import write_prompt_examples_jsonl
from learning_to_reset.pipeline import prepare_countdown_examples, prepare_sft_examples


def _train_validation_split(items, *, val_ratio: float):
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    items = tuple(items)
    train_end = int(len(items) * (1 - val_ratio))
    return items[:train_end], items[train_end:]


def export_paper_prepared_datasets(
    *,
    trace_path,
    countdown_train_path,
    countdown_eval_path,
    output_dir,
    sft_val_ratio: float = 0.1,
    countdown_val_ratio: float = 0.1,
    hard_mine_ratio: float = 0.2,
    allow_clean: bool = True,
    include_recovery_examples: bool = False,
    recovery_repeat: int = 1,
    require_recovery_target_correct: bool = False,
) -> Dict[str, Any]:
    """Prepare artifacts using distinct Countdown train/eval source files."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    trace_examples = prepare_sft_examples(
        load_trace_records(trace_path),
        include_recovery_examples=include_recovery_examples,
        recovery_repeat=recovery_repeat,
        require_recovery_target_correct=require_recovery_target_correct,
    )
    sft_train, sft_validation = _train_validation_split(trace_examples, val_ratio=sft_val_ratio)

    countdown_train_examples = prepare_countdown_examples(
        load_countdown_samples(countdown_train_path),
        allow_clean=allow_clean,
    )
    countdown_eval_samples = load_countdown_samples(countdown_eval_path)
    countdown_eval_examples = prepare_countdown_examples(
        countdown_eval_samples,
        allow_clean=allow_clean,
    )
    hard_countdown_train_samples = filter_hard_countdown_samples(
        load_countdown_samples(countdown_train_path)
    )
    hard_countdown_train_examples = prepare_countdown_examples(
        hard_countdown_train_samples,
        allow_clean=allow_clean,
    )
    countdown_hard_eval_examples = prepare_countdown_examples(
        filter_hard_countdown_samples(countdown_eval_samples),
        allow_clean=allow_clean,
    )
    countdown_train, countdown_validation = _train_validation_split(
        countdown_train_examples,
        val_ratio=countdown_val_ratio,
    )
    hard_countdown_train, hard_countdown_mine = _train_validation_split(
        hard_countdown_train_examples,
        val_ratio=hard_mine_ratio,
    )

    write_prompt_examples_jsonl(sft_train, output_root / "sft-train.jsonl")
    write_prompt_examples_jsonl(sft_validation, output_root / "sft-validation.jsonl")
    write_prompt_examples_jsonl(countdown_train, output_root / "countdown-train.jsonl")
    write_prompt_examples_jsonl(countdown_validation, output_root / "countdown-validation.jsonl")
    write_prompt_examples_jsonl(
        hard_countdown_train, output_root / "countdown-train-hard.jsonl"
    )
    write_prompt_examples_jsonl(
        hard_countdown_mine, output_root / "countdown-mine-hard.jsonl"
    )
    write_prompt_examples_jsonl(countdown_eval_examples, output_root / "countdown-test.jsonl")
    write_prompt_examples_jsonl(countdown_hard_eval_examples, output_root / "countdown-test-hard.jsonl")

    manifest = {
        "sft": {
            "train": len(sft_train),
            "validation": len(sft_validation),
        },
        "countdown": {
            "train": len(countdown_train),
            "validation": len(countdown_validation),
            "test": len(countdown_eval_examples),
            "test_hard": len(countdown_hard_eval_examples),
            "train_hard": len(hard_countdown_train),
            "mine_hard": len(hard_countdown_mine),
        },
        "allow_clean": allow_clean,
        "include_recovery_examples": include_recovery_examples,
        "recovery_repeat": recovery_repeat,
        "require_recovery_target_correct": require_recovery_target_correct,
        "sft_val_ratio": sft_val_ratio,
        "countdown_val_ratio": countdown_val_ratio,
        "hard_mine_ratio": hard_mine_ratio,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest

"""Export utilities that turn loaded data into prepared JSONL artifacts."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Union

from learning_to_reset.data import PathLike, load_countdown_samples, load_trace_records
from learning_to_reset.pipeline import prepare_countdown_examples, prepare_sft_examples, split_sequence
from learning_to_reset.prompts import PromptExample


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def serialize_prompt_example(example: PromptExample) -> Dict[str, Any]:
    """Convert a prompt example into a JSON-safe payload."""

    return {
        "prompt": example.prompt,
        "response": example.response,
        "metadata": _jsonify(example.metadata),
    }


def write_prompt_examples_jsonl(examples, path: Union[str, Path]) -> None:
    """Write a sequence of prompt examples to JSONL."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(serialize_prompt_example(example), ensure_ascii=True) for example in examples]
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    destination.write_text(payload, encoding="utf-8")


def prepare_trace_split(
    path: PathLike,
    *,
    train_ratio: float,
    val_ratio: float,
):
    """Load trace records and turn them into split SFT examples."""

    records = load_trace_records(path)
    examples = prepare_sft_examples(records)
    return split_sequence(examples, train_ratio=train_ratio, val_ratio=val_ratio)


def prepare_countdown_split(
    path: PathLike,
    *,
    train_ratio: float,
    val_ratio: float,
    allow_clean: bool,
):
    """Load Countdown samples and turn them into split prompt-only examples."""

    samples = load_countdown_samples(path)
    examples = prepare_countdown_examples(samples, allow_clean=allow_clean)
    return split_sequence(examples, train_ratio=train_ratio, val_ratio=val_ratio)


def _count_split(split) -> Dict[str, int]:
    return {
        "train": len(split.train),
        "validation": len(split.validation),
        "test": len(split.test),
    }


def export_prepared_datasets(
    *,
    trace_path: PathLike,
    countdown_path: PathLike,
    output_dir: PathLike,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    allow_clean_eval: bool = False,
) -> Dict[str, Mapping[str, int]]:
    """Prepare and export split datasets plus a manifest."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    sft_split = prepare_trace_split(
        trace_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    countdown_split = prepare_countdown_split(
        countdown_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        allow_clean=allow_clean_eval,
    )

    write_prompt_examples_jsonl(sft_split.train, output_root / "sft-train.jsonl")
    write_prompt_examples_jsonl(sft_split.validation, output_root / "sft-validation.jsonl")
    write_prompt_examples_jsonl(sft_split.test, output_root / "sft-test.jsonl")
    write_prompt_examples_jsonl(countdown_split.train, output_root / "countdown-train.jsonl")
    write_prompt_examples_jsonl(countdown_split.validation, output_root / "countdown-validation.jsonl")
    write_prompt_examples_jsonl(countdown_split.test, output_root / "countdown-test.jsonl")

    manifest = {
        "sft": _count_split(sft_split),
        "countdown": _count_split(countdown_split),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "allow_clean_eval": allow_clean_eval,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(_jsonify(manifest), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest

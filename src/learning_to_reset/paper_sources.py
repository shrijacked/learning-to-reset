"""Paper-aligned source adapters for external datasets and models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from learning_to_reset.data import render_countdown_question


DEFAULT_TARGET_MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEFAULT_TRACE_SOURCE_MODEL_ID = "obiwan96/qwen-cd-100"
DEFAULT_COUNTDOWN_TRAIN_DATASET_ID = "obiwan96/countdown-env-train"
DEFAULT_COUNTDOWN_EVAL_DATASET_ID = "obiwan96/countdown-env-eval"
DEFAULT_REFERENCE_TRACE_DATASET_ID = "obiwan96/owm-cog-behaviors"

BEHAVIOR_QUERY_PATTERN = re.compile(r"User:\s*(.*?)\nAssistant:", re.DOTALL)
COUNTDOWN_USER_PATTERN = re.compile(
    r"Using the numbers\s*\[(.*?)\],\s*create an equation that equals\s*(-?\d+)\.?",
    re.IGNORECASE | re.DOTALL,
)


def _require_datasets():
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - runtime only
        raise RuntimeError(
            "datasets is required to fetch the external paper assets. "
            "Install it in the active environment first."
        ) from exc
    return load_dataset


def write_jsonl_records(records: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """Write JSONL records in a stable ASCII-safe format."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(dict(record), ensure_ascii=True) for record in records)
    if payload:
        payload += "\n"
    destination.write_text(payload, encoding="utf-8")


def extract_behavior_question(query: str) -> str:
    """Extract the user question from the upstream behavior-trace prompt format."""

    matches = BEHAVIOR_QUERY_PATTERN.findall(query)
    if not matches:
        raise ValueError("Could not extract a user question from the behavior dataset query.")
    return matches[-1].strip()


def parse_countdown_user_prompt(prompt_text: str) -> Tuple[Tuple[int, ...], int]:
    """Parse numbers and target from the Countdown user message."""

    match = COUNTDOWN_USER_PATTERN.search(prompt_text.strip())
    if match is None:
        raise ValueError("Could not parse the Countdown user prompt.")

    numbers = tuple(int(part.strip()) for part in match.group(1).split(",") if part.strip())
    target = int(match.group(2))
    return numbers, target


def _extract_user_prompt(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in messages:
        if str(message.get("role", "")).lower() == "user":
            return str(message.get("content", "")).strip()
    raise ValueError("The Countdown prompt row does not contain a user message.")


def expand_countdown_hf_row(
    row: Mapping[str, Any],
    *,
    dataset_id: str,
    split: str,
    row_index: int,
) -> Tuple[Dict[str, Any], ...]:
    """Flatten one upstream Countdown row into local JSONL records."""

    user_prompt = _extract_user_prompt(row["prompt"])
    try:
        prompt_numbers, prompt_target = parse_countdown_user_prompt(user_prompt)
    except ValueError:
        prompt_numbers = ()
        prompt_target = 0

    metadata = row["metadata"]
    records = []
    for sample_index, (numbers, target) in enumerate(zip(metadata["numbers"], metadata["target"])):
        normalized_numbers = [int(number) for number in numbers]
        normalized_target = int(target)
        if (
            sample_index == 0
            and tuple(normalized_numbers) == prompt_numbers
            and normalized_target == prompt_target
        ):
            question = user_prompt
        else:
            question = render_countdown_question(normalized_numbers, normalized_target)

        records.append(
            {
                "source_id": f"{dataset_id}:{split}:{row_index}:{sample_index}",
                "numbers": normalized_numbers,
                "target": normalized_target,
                "question": question,
                "metadata": {
                    "source_dataset": dataset_id,
                    "source_split": split,
                    "row_index": row_index,
                    "sample_index": sample_index,
                    "T_max": metadata.get("T_max"),
                },
            }
        )
    return tuple(records)


def fetch_countdown_source_records(
    *,
    dataset_id: str,
    split: str,
    output_path: str | Path,
    max_rows: int | None = None,
    max_samples: int | None = None,
) -> Dict[str, Any]:
    """Fetch and flatten an upstream Countdown dataset into local JSONL."""

    load_dataset = _require_datasets()
    dataset = load_dataset(dataset_id, split=split, streaming=True)

    records: List[Dict[str, Any]] = []
    row_count = 0
    for row_index, row in enumerate(dataset):
        if max_rows is not None and row_index >= max_rows:
            break

        row_records = expand_countdown_hf_row(
            row,
            dataset_id=dataset_id,
            split=split,
            row_index=row_index,
        )
        for record in row_records:
            if max_samples is not None and len(records) >= max_samples:
                break
            records.append(record)
        row_count += 1
        if max_samples is not None and len(records) >= max_samples:
            break

    write_jsonl_records(records, output_path)
    return {
        "dataset_id": dataset_id,
        "split": split,
        "rows_read": row_count,
        "samples_written": len(records),
        "output_path": str(output_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and flatten the paper-aligned source datasets.")
    parser.add_argument(
        "--countdown-train-dataset",
        default=DEFAULT_COUNTDOWN_TRAIN_DATASET_ID,
        help="Countdown train dataset id.",
    )
    parser.add_argument(
        "--countdown-eval-dataset",
        default=DEFAULT_COUNTDOWN_EVAL_DATASET_ID,
        help="Countdown eval dataset id.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for local JSONL source files.")
    parser.add_argument("--max-train-rows", type=int, help="Optional train row limit for quick experiments.")
    parser.add_argument("--max-eval-rows", type=int, help="Optional eval row limit for quick experiments.")
    parser.add_argument("--max-train-samples", type=int, help="Optional train sample limit.")
    parser.add_argument("--max-eval-samples", type=int, help="Optional eval sample limit.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_summary = fetch_countdown_source_records(
        dataset_id=args.countdown_train_dataset,
        split="train",
        output_path=output_dir / "countdown-train.jsonl",
        max_rows=args.max_train_rows,
        max_samples=args.max_train_samples,
    )
    eval_summary = fetch_countdown_source_records(
        dataset_id=args.countdown_eval_dataset,
        split="eval",
        output_path=output_dir / "countdown-eval.jsonl",
        max_rows=args.max_eval_rows,
        max_samples=args.max_eval_samples,
    )

    summary = {
        "target_model_id": DEFAULT_TARGET_MODEL_ID,
        "trace_source_model_id": DEFAULT_TRACE_SOURCE_MODEL_ID,
        "reference_trace_dataset_id": DEFAULT_REFERENCE_TRACE_DATASET_ID,
        "countdown_train": train_summary,
        "countdown_eval": eval_summary,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

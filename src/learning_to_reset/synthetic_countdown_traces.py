"""Task-aligned synthetic Countdown traces used as a local fallback trace source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from learning_to_reset.countdown_solver import solve_countdown
from learning_to_reset.countdown_verifier import verify_countdown_expression
from learning_to_reset.data import CountdownSample, load_countdown_samples
from learning_to_reset.paper_sources import write_jsonl_records


def build_positive_trace_record(
    sample: CountdownSample,
    *,
    solution_expression: str,
) -> Dict[str, Any]:
    """Create a productive tagged trace for one Countdown sample."""

    return {
        "source_id": f"{sample.source_id or 'countdown'}:synthetic-positive",
        "problem": sample.question,
        "raw_trace": (
            "<think>\n"
            "I can solve this by combining the available numbers into one expression "
            f"that reaches {sample.target}. A valid construction is {solution_expression}.\n"
            "</think>\n"
            "<answer>\n"
            f"{solution_expression}\n"
            "</answer>"
        ),
        "is_correct": True,
        "metadata": {
            "source": "synthetic_countdown_solver",
            "numbers": list(sample.numbers),
            "target": sample.target,
            "solution_expression": solution_expression,
        },
    }


def build_negative_expression(sample: CountdownSample) -> str:
    """Pick a simple legal expression that stays off target for reset supervision."""

    candidate_expressions = [str(int(number)) for number in sample.numbers]
    if len(sample.numbers) >= 2:
        left = int(sample.numbers[0])
        right = int(sample.numbers[1])
        candidate_expressions.extend(
            [
                f"({left} + {right})",
                f"({left} - {right})",
                f"({right} - {left})",
                f"({left} * {right})",
            ]
        )
        if right != 0:
            candidate_expressions.append(f"({left} / {right})")
        if left != 0:
            candidate_expressions.append(f"({right} / {left})")

    for expression in candidate_expressions:
        verification = verify_countdown_expression(
            expression,
            numbers=sample.numbers,
            target=sample.target,
        )
        if verification.is_valid and not verification.reaches_target:
            return expression

    raise ValueError(
        f"Could not build a synthetic incorrect expression for sample {sample.source_id!r}."
    )


def build_negative_trace_record(sample: CountdownSample) -> Dict[str, Any]:
    """Create an unproductive tagged trace that the curator can turn into `<clean>`."""

    incorrect_expression = build_negative_expression(sample)
    return {
        "source_id": f"{sample.source_id or 'countdown'}:synthetic-negative",
        "problem": sample.question,
        "raw_trace": (
            "<think>\n"
            "I try a straightforward combination first, but it does not actually reach "
            f"the target. My tentative expression is {incorrect_expression}.\n"
            "</think>\n"
            "<answer>\n"
            f"{incorrect_expression}\n"
            "</answer>"
        ),
        "is_correct": False,
        "metadata": {
            "source": "synthetic_countdown_solver",
            "numbers": list(sample.numbers),
            "target": sample.target,
            "incorrect_expression": incorrect_expression,
        },
    }


def build_synthetic_trace_records(
    samples: Iterable[CountdownSample],
    *,
    include_negative: bool = True,
) -> Tuple[List[Dict[str, Any]], int]:
    """Build a paired synthetic trace corpus from Countdown samples."""

    records: List[Dict[str, Any]] = []
    skipped = 0
    for sample in samples:
        solution_expression = sample.solution or solve_countdown(sample.numbers, sample.target)
        if solution_expression is None:
            skipped += 1
            continue

        records.append(
            build_positive_trace_record(
                sample,
                solution_expression=solution_expression,
            )
        )
        if include_negative:
            records.append(build_negative_trace_record(sample))
    return records, skipped


def generate_synthetic_trace_corpus(
    *,
    countdown_path: str | Path,
    output_path: str | Path,
    max_samples: int | None = None,
    include_negative: bool = True,
) -> Dict[str, Any]:
    """Write a synthetic Countdown-aligned trace corpus to JSONL."""

    samples = list(load_countdown_samples(countdown_path))
    if max_samples is not None:
        samples = samples[:max_samples]

    records, skipped = build_synthetic_trace_records(
        samples,
        include_negative=include_negative,
    )
    write_jsonl_records(records, output_path)

    summary = {
        "countdown_examples": len(samples),
        "records_written": len(records),
        "skipped_unsolved": skipped,
        "include_negative": include_negative,
        "output_path": str(output_path),
    }
    Path(output_path).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Countdown-aligned traces from local source questions."
    )
    parser.add_argument("--countdown", required=True, help="Countdown JSON/JSONL file.")
    parser.add_argument("--output-path", required=True, help="Synthetic trace JSONL path.")
    parser.add_argument("--max-samples", type=int, help="Optional sample cap.")
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Write only productive traces and skip synthetic negative traces.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = generate_synthetic_trace_corpus(
        countdown_path=args.countdown,
        output_path=args.output_path,
        max_samples=args.max_samples,
        include_negative=not args.positive_only,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""Generate additional solvable Countdown samples offline from local references."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import random
from typing import Dict, List, Optional, Sequence, Tuple

from learning_to_reset.countdown_slices import (
    can_reach_with_add_sub_only,
    expression_uses_multiplication_or_division,
)
from learning_to_reset.data import CountdownSample, load_countdown_samples, render_countdown_question
from learning_to_reset.paper_sources import write_jsonl_records


@dataclass(frozen=True)
class CountdownSamplingProfile:
    """Empirical sampling profile inferred from local Countdown references."""

    arities: Tuple[int, ...]
    number_pool: Tuple[int, ...]
    min_target: int
    max_target: int


@dataclass(frozen=True)
class CandidateExpression:
    """One partial arithmetic expression used while sampling a new puzzle."""

    value: Fraction
    expression: str


def build_sampling_profile(samples: Sequence[CountdownSample]) -> CountdownSamplingProfile:
    """Infer a lightweight local sampling profile from reference Countdown prompts."""

    if not samples:
        raise ValueError("At least one reference sample is required.")

    arities = tuple(len(sample.numbers) for sample in samples)
    number_pool = tuple(int(number) for sample in samples for number in sample.numbers)
    return CountdownSamplingProfile(
        arities=arities,
        number_pool=number_pool,
        min_target=min(int(sample.target) for sample in samples),
        max_target=max(int(sample.target) for sample in samples),
    )


def _combine_candidates(
    left: CandidateExpression,
    right: CandidateExpression,
) -> Tuple[CandidateExpression, ...]:
    candidates = [
        CandidateExpression(left.value + right.value, f"({left.expression} + {right.expression})"),
        CandidateExpression(left.value - right.value, f"({left.expression} - {right.expression})"),
        CandidateExpression(right.value - left.value, f"({right.expression} - {left.expression})"),
        CandidateExpression(left.value * right.value, f"({left.expression} * {right.expression})"),
    ]
    if right.value != 0:
        candidates.append(
            CandidateExpression(left.value / right.value, f"({left.expression} / {right.expression})")
        )
    if left.value != 0:
        candidates.append(
            CandidateExpression(right.value / left.value, f"({right.expression} / {left.expression})")
        )

    deduplicated: List[CandidateExpression] = []
    seen = set()
    for candidate in candidates:
        key = (candidate.value.numerator, candidate.value.denominator, candidate.expression)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(candidate)
    return tuple(deduplicated)


def _sample_target_expression(
    numbers: Sequence[int],
    *,
    profile: CountdownSamplingProfile,
    rng: random.Random,
    max_attempts: int = 256,
) -> Tuple[int, str]:
    """Build one solvable integer target and witness expression from chosen numbers."""

    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")

    for _ in range(max_attempts):
        items = [
            CandidateExpression(value=Fraction(int(number)), expression=str(int(number)))
            for number in numbers
        ]

        while len(items) > 1:
            left_index, right_index = sorted(rng.sample(range(len(items)), 2))
            left = items.pop(right_index)
            right = items.pop(left_index)
            merged = rng.choice(_combine_candidates(left, right))
            items.append(merged)

        candidate = items[0]
        if candidate.value.denominator != 1:
            continue

        target = int(candidate.value)
        if target < profile.min_target or target > profile.max_target:
            continue

        return target, candidate.expression

    raise RuntimeError("Could not sample a solvable integer Countdown target within the profile range.")


def _requires_multiplication_or_division(
    *,
    numbers: Sequence[int],
    target: int,
    expression: str,
) -> bool:
    return (
        expression_uses_multiplication_or_division(expression)
        and not can_reach_with_add_sub_only(numbers, target)
    )


def generate_synthetic_countdown_dataset(
    *,
    reference_path: str | Path,
    output_path: str | Path,
    num_samples: int,
    seed: int = 0,
    require_hard: bool = False,
) -> Dict[str, object]:
    """Generate an offline Countdown dataset shaped like the local reference slice."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    reference_samples = load_countdown_samples(reference_path)
    profile = build_sampling_profile(reference_samples)
    rng = random.Random(seed)

    records = []
    seen = set()
    max_total_attempts = max(num_samples * 64, 256)
    attempts = 0

    while len(records) < num_samples and attempts < max_total_attempts:
        attempts += 1
        arity = rng.choice(profile.arities)
        numbers = tuple(int(rng.choice(profile.number_pool)) for _ in range(arity))

        try:
            target, seed_expression = _sample_target_expression(
                numbers,
                profile=profile,
                rng=rng,
            )
        except RuntimeError:
            continue
        if require_hard and not _requires_multiplication_or_division(
            numbers=numbers,
            target=target,
            expression=seed_expression,
        ):
            continue

        key = (numbers, target)
        if key in seen:
            continue
        seen.add(key)

        records.append(
            {
                "source_id": f"synthetic-countdown:{len(records)}",
                "numbers": list(numbers),
                "target": target,
                "question": render_countdown_question(numbers, target),
                "metadata": {
                    "source": "synthetic_countdown_dataset",
                    "seed_expression": seed_expression,
                    "reference_path": str(reference_path),
                    "difficulty": "hard" if require_hard else "mixed",
                    "requires_mul_div": require_hard,
                },
            }
        )

    if len(records) < num_samples:
        raise RuntimeError(
            "Could not generate enough unique solvable Countdown samples from the local profile."
        )

    write_jsonl_records(records, output_path)
    summary = {
        "reference_examples": len(reference_samples),
        "samples_written": len(records),
        "arity_options": sorted(set(profile.arities)),
        "target_range": [profile.min_target, profile.max_target],
        "seed": seed,
        "require_hard": require_hard,
        "output_path": str(output_path),
    }
    Path(output_path).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate additional local Countdown samples from a reference slice."
    )
    parser.add_argument("--reference-path", required=True, help="Reference Countdown JSON/JSONL file.")
    parser.add_argument("--output-path", required=True, help="Output JSONL for generated samples.")
    parser.add_argument("--num-samples", required=True, type=int, help="How many samples to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic generation.")
    parser.add_argument(
        "--require-hard",
        action="store_true",
        help="Only write samples that require multiplication or division under the local hard-slice filter.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = generate_synthetic_countdown_dataset(
        reference_path=args.reference_path,
        output_path=args.output_path,
        num_samples=args.num_samples,
        seed=args.seed,
        require_hard=args.require_hard,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

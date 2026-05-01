"""Preparation utilities that bridge loaded records into future trainers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import re
from typing import Any, Generic, Optional, Sequence, Tuple, TypeVar

from learning_to_reset.countdown_verifier import VerificationResult, score_countdown_response
from learning_to_reset.data import CountdownSample, TraceRecord
from learning_to_reset.prompts import (
    PromptExample,
    build_countdown_prompt,
    build_reasoning_prompt,
    build_sft_training_example,
)


_ARITHMETIC_CLAIM_PATTERN = re.compile(
    r"(?<![\d/])(-?\d+(?:/\d+)?)\s*([+\-*/])\s*(-?\d+(?:/\d+)?)\s*=\s*"
    r"(-?\d+(?:/\d+)?)(?![\d/])"
)


def _deep_verify_recovery_text(text: str) -> bool:
    """Re-check every plain `a op b = c` claim in a recovery body with `Fraction`.

    Defense in depth on top of the answer-block target check: if any inline
    arithmetic claim does not actually compute as written, drop the trace.
    Parenthesized claims like `((9 * 11) - (12 + 17)) = 70` are intentionally
    skipped because the answer-block verification already proves the outermost
    expression. This filter is meant to catch lying intermediate substeps,
    which is the failure mode mined from the 1/32 hard-holdout baseline.
    """

    for match in _ARITHMETIC_CLAIM_PATTERN.finditer(text):
        try:
            left = Fraction(match.group(1))
            right = Fraction(match.group(3))
            claimed = Fraction(match.group(4))
        except (ValueError, ZeroDivisionError):
            continue

        op = match.group(2)
        if op == "+":
            actual = left + right
        elif op == "-":
            actual = left - right
        elif op == "*":
            actual = left * right
        elif op == "/":
            if right == 0:
                return False
            actual = left / right
        else:
            continue

        if actual != claimed:
            return False
    return True


T = TypeVar("T")


@dataclass(frozen=True)
class DatasetSplit(Generic[T]):
    """A deterministic train/validation/test partition."""

    train: Tuple[T, ...]
    validation: Tuple[T, ...]
    test: Tuple[T, ...]


@dataclass(frozen=True)
class PromptBatch:
    """A batch of prompt examples for later trainer integration."""

    examples: Tuple[PromptExample, ...]


def split_sequence(
    items: Sequence[T],
    *,
    train_ratio: float,
    val_ratio: float,
) -> DatasetSplit[T]:
    """Split a sequence deterministically while preserving order."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    items_tuple = tuple(items)
    total = len(items_tuple)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return DatasetSplit(
        train=items_tuple[:train_end],
        validation=items_tuple[train_end:val_end],
        test=items_tuple[val_end:],
    )


def _coerce_countdown_numbers(value: Any) -> Tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    if isinstance(value, str):
        parts = [part for part in re.split(r"[\s,]+", value.strip()) if part]
        return tuple(int(part) for part in parts)
    raise ValueError(f"Could not coerce Countdown numbers: {value!r}")


def _countdown_sample_from_record_metadata(record: TraceRecord) -> Optional[CountdownSample]:
    numbers = record.metadata.get("numbers")
    target = record.metadata.get("target")
    if numbers is None or target in (None, ""):
        return None

    try:
        return CountdownSample(
            numbers=_coerce_countdown_numbers(numbers),
            target=int(target),
            question=record.problem,
            source_id=record.source_id,
        )
    except (TypeError, ValueError):
        return None


def _verify_recovery_response(
    record: TraceRecord,
    recovery_response: str,
) -> Optional[VerificationResult]:
    sample = _countdown_sample_from_record_metadata(record)
    if sample is None:
        return None

    verification = score_countdown_response(recovery_response, sample)
    if not verification.is_valid or not verification.reaches_target:
        return None
    if not _deep_verify_recovery_text(recovery_response):
        return None
    return verification


def prepare_retry_recovery_examples(
    records: Sequence[TraceRecord],
    *,
    repeat: int = 1,
    require_target_correct: bool = False,
) -> Tuple[PromptExample, ...]:
    """Build retry-stage recovery examples for records that carry explicit recovery targets."""

    if repeat <= 0:
        raise ValueError("repeat must be positive.")

    examples = []
    for record in records:
        recovery_response = record.metadata.get("recovery_response")
        if record.is_correct or not recovery_response:
            continue

        verification = None
        if require_target_correct:
            verification = _verify_recovery_response(record, str(recovery_response))
            if verification is None:
                continue

        metadata = {
            "source_id": record.source_id,
            "is_correct": True,
            "uses_clean": False,
            "stage": "retry-recovery",
        }
        if verification is not None:
            metadata.update(
                {
                    "recovery_expression": verification.expression,
                    "recovery_value": str(verification.value),
                    "recovery_verified": True,
                }
            )

        for _ in range(repeat):
            examples.append(
                PromptExample(
                    prompt=build_reasoning_prompt(record.problem, allow_clean=False),
                    response=str(recovery_response).strip(),
                    metadata=dict(metadata),
                )
            )
    return tuple(examples)


def prepare_sft_examples(
    records: Sequence[TraceRecord],
    *,
    include_recovery_examples: bool = False,
    recovery_repeat: int = 1,
    require_recovery_target_correct: bool = False,
    exclude_bootstrap_negatives: bool = False,
) -> Tuple[PromptExample, ...]:
    """Convert trace records into prompt/response examples for SFT."""

    filtered_records = tuple(
        record
        for record in records
        if not (
            exclude_bootstrap_negatives
            and record.metadata.get("bootstrap_kind") == "think_only_negative"
        )
    )

    examples = [build_sft_training_example(record) for record in filtered_records]
    if include_recovery_examples:
        examples.extend(
            prepare_retry_recovery_examples(
                filtered_records,
                repeat=recovery_repeat,
                require_target_correct=require_recovery_target_correct,
            )
        )
    return tuple(examples)


def summarize_sft_example_mix(
    examples: Sequence[PromptExample],
) -> dict[str, object]:
    """Summarize prepared SFT examples by source/style/stage and clean usage."""

    source_counter: Counter[str] = Counter()
    stage_counter: Counter[str] = Counter()
    recovery_style_counter: Counter[str] = Counter()
    bootstrap_kind_counter: Counter[str] = Counter()
    uses_clean_counter: Counter[str] = Counter()

    for example in examples:
        metadata = example.metadata or {}
        source_counter[str(metadata.get("source", "unknown"))] += 1
        stage_counter[str(metadata.get("stage", "unknown"))] += 1
        recovery_style = metadata.get("recovery_style")
        if recovery_style not in (None, ""):
            recovery_style_counter[str(recovery_style)] += 1
        bootstrap_kind = metadata.get("bootstrap_kind")
        if bootstrap_kind not in (None, ""):
            bootstrap_kind_counter[str(bootstrap_kind)] += 1
        uses_clean_counter["true" if bool(metadata.get("uses_clean")) else "false"] += 1

    return {
        "total_examples": len(examples),
        "by_source": dict(sorted(source_counter.items())),
        "by_stage": dict(sorted(stage_counter.items())),
        "by_recovery_style": dict(sorted(recovery_style_counter.items())),
        "by_bootstrap_kind": dict(sorted(bootstrap_kind_counter.items())),
        "by_uses_clean": dict(sorted(uses_clean_counter.items())),
    }


def prepare_countdown_examples(
    samples: Sequence[CountdownSample],
    *,
    allow_clean: bool,
) -> Tuple[PromptExample, ...]:
    """Convert Countdown samples into prompt-only evaluation examples."""

    examples = []
    for sample in samples:
        examples.append(
            PromptExample(
                prompt=build_countdown_prompt(sample, allow_clean=allow_clean),
                response="",
                metadata={
                    "source_id": sample.source_id,
                    "numbers": sample.numbers,
                    "target": sample.target,
                    "solution": sample.solution,
                    "question": sample.question,
                },
            )
        )
    return tuple(examples)


def batch_prompt_examples(
    examples: Sequence[PromptExample],
    *,
    batch_size: int,
) -> Tuple[PromptBatch, ...]:
    """Chunk prompt examples into deterministic batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    items = tuple(examples)
    batches = []
    for start in range(0, len(items), batch_size):
        batches.append(PromptBatch(examples=items[start : start + batch_size]))
    return tuple(batches)

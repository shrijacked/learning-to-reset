"""Preparation utilities that bridge loaded records into future trainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Sequence, Tuple, TypeVar

from learning_to_reset.data import CountdownSample, TraceRecord
from learning_to_reset.prompts import (
    PromptExample,
    build_countdown_prompt,
    build_reasoning_prompt,
    build_sft_training_example,
)


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


def prepare_retry_recovery_examples(
    records: Sequence[TraceRecord],
    *,
    repeat: int = 1,
) -> Tuple[PromptExample, ...]:
    """Build retry-stage recovery examples for records that carry explicit recovery targets."""

    if repeat <= 0:
        raise ValueError("repeat must be positive.")

    examples = []
    for record in records:
        recovery_response = record.metadata.get("recovery_response")
        if record.is_correct or not recovery_response:
            continue

        for _ in range(repeat):
            examples.append(
                PromptExample(
                    prompt=build_reasoning_prompt(record.problem, allow_clean=False),
                    response=str(recovery_response).strip(),
                    metadata={
                        "source_id": record.source_id,
                        "is_correct": True,
                        "uses_clean": False,
                        "stage": "retry-recovery",
                    },
                )
            )
    return tuple(examples)


def prepare_sft_examples(
    records: Sequence[TraceRecord],
    *,
    include_recovery_examples: bool = False,
    recovery_repeat: int = 1,
) -> Tuple[PromptExample, ...]:
    """Convert trace records into prompt/response examples for SFT."""

    examples = [build_sft_training_example(record) for record in records]
    if include_recovery_examples:
        examples.extend(
            prepare_retry_recovery_examples(
                records,
                repeat=recovery_repeat,
            )
        )
    return tuple(examples)


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

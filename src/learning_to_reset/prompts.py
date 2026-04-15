"""Prompt builders that connect loaded data to the reset-aware baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from learning_to_reset.data import CountdownSample, TraceRecord
from learning_to_reset.trace_curation import curate_trace


DEFAULT_BASE_INSTRUCTIONS = (
    "Think inside <think> tags and provide the final expression inside <answer> tags."
)
DEFAULT_CLEAN_INSTRUCTIONS = (
    "If your search becomes confusing or unproductive, explain the reset and emit <clean>."
)


@dataclass(frozen=True)
class PromptExample:
    """A generic prompt/response pair ready for later training or evaluation code."""

    prompt: str
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_reasoning_prompt(
    question: str,
    *,
    allow_clean: bool,
    base_instructions: str = DEFAULT_BASE_INSTRUCTIONS,
    clean_instructions: str = DEFAULT_CLEAN_INSTRUCTIONS,
) -> str:
    parts = [base_instructions.strip()]
    if allow_clean:
        parts.append(clean_instructions.strip())
    parts.append(f"Question: {question.strip()}")
    return "\n\n".join(part for part in parts if part)


def build_sft_training_example(
    record: TraceRecord,
    *,
    base_instructions: str = DEFAULT_BASE_INSTRUCTIONS,
    clean_instructions: str = DEFAULT_CLEAN_INSTRUCTIONS,
) -> PromptExample:
    curated = curate_trace(
        raw_trace=record.raw_trace,
        is_correct=record.is_correct,
        base_instructions=base_instructions,
        clean_instructions=clean_instructions,
    )
    prompt = build_reasoning_prompt(
        record.problem,
        allow_clean=True,
        base_instructions=base_instructions,
        clean_instructions=clean_instructions,
    )
    metadata = {
        "source_id": record.source_id,
        "is_correct": record.is_correct,
        "uses_clean": curated.uses_clean,
    }
    return PromptExample(prompt=prompt, response=curated.response, metadata=metadata)


def build_countdown_prompt(
    sample: CountdownSample,
    *,
    allow_clean: bool,
    base_instructions: str = DEFAULT_BASE_INSTRUCTIONS,
    clean_instructions: str = DEFAULT_CLEAN_INSTRUCTIONS,
) -> str:
    return build_reasoning_prompt(
        sample.question,
        allow_clean=allow_clean,
        base_instructions=base_instructions,
        clean_instructions=clean_instructions,
    )

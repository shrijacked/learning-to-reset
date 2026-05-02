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
QUESTION_PREFIX = "Question:"


@dataclass(frozen=True)
class PromptExample:
    """A generic prompt/response pair ready for later training or evaluation code."""

    prompt: str
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReasoningPromptParts:
    """Structured parts extracted from a reasoning prompt."""

    base_instructions: str
    clean_instructions: str | None
    question: str


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
    parts.append(f"{QUESTION_PREFIX} {question.strip()}")
    return "\n\n".join(part for part in parts if part)


def parse_reasoning_prompt(prompt: str) -> ReasoningPromptParts:
    """Recover instructions and question text from a prepared prompt."""

    text = prompt.strip()
    question_marker = f"\n\n{QUESTION_PREFIX}"
    if question_marker in text:
        prefix, question_block = text.rsplit(question_marker, 1)
    elif text.startswith(QUESTION_PREFIX):
        prefix = ""
        question_block = text[len(QUESTION_PREFIX) :]
    else:
        raise ValueError("Prompt does not contain a recoverable question block.")

    instruction_parts = [part.strip() for part in prefix.split("\n\n") if part.strip()]
    return ReasoningPromptParts(
        base_instructions=instruction_parts[0] if instruction_parts else "",
        clean_instructions="\n\n".join(instruction_parts[1:]) or None,
        question=question_block.strip(),
    )


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
        allow_missing_answer_for_incorrect=(
            record.metadata.get("bootstrap_kind") == "think_only_negative"
        ),
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
        "stage": "base-trace",
    }
    for key in ("source", "recovery_style", "bootstrap_kind", "bootstrap_source_id"):
        value = record.metadata.get(key)
        if value not in (None, ""):
            metadata[key] = value
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

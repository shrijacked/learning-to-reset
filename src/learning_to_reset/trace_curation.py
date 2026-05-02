"""SFT trace curation utilities for context-reset training."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional


THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
DEFAULT_CLEAN_SIGNAL = (
    "I realize my search is becoming confusing, so I should clean my context."
)


@dataclass(frozen=True)
class NormalizedTrace:
    """Normalized expert trace with one think block and the final answer block."""

    think_text: str
    answer_text: Optional[str] = None

    def as_response(self) -> str:
        """Render the trace back into the project's tagged response format."""
        parts = [f"<think>\n{self.think_text}\n</think>"]
        if self.answer_text is not None:
            parts.append(f"<answer>\n{self.answer_text}\n</answer>")
        return "\n".join(parts)


@dataclass(frozen=True)
class CuratedTrace:
    """Final curated SFT example."""

    instructions: str
    response: str
    uses_clean: bool
    normalized_trace: NormalizedTrace


def _normalize_block_text(block: str) -> str:
    lines = [line.strip() for line in block.splitlines()]
    return "\n".join(line for line in lines if line)


def _extract_blocks(pattern: re.Pattern[str], raw_trace: str) -> List[str]:
    return [_normalize_block_text(block) for block in pattern.findall(raw_trace)]


def normalize_trace(raw_trace: str) -> NormalizedTrace:
    """Collapse a raw trace into one think block plus the final answer.

    Repeated think blocks are merged, only the final answer is retained, and
    everything outside the tagged blocks is discarded.
    """

    think_blocks = _extract_blocks(THINK_PATTERN, raw_trace)
    if not think_blocks:
        raise ValueError("A trace must contain at least one <think> block.")

    answer_blocks = _extract_blocks(ANSWER_PATTERN, raw_trace)

    think_text = "\n\n".join(block for block in think_blocks if block)
    answer_text = answer_blocks[-1] if answer_blocks else None
    return NormalizedTrace(think_text=think_text, answer_text=answer_text)


def _combine_instructions(base_instructions: str, clean_instructions: str) -> str:
    parts = [base_instructions.strip(), clean_instructions.strip()]
    return "\n\n".join(part for part in parts if part)


def curate_trace(
    *,
    raw_trace: str,
    is_correct: bool,
    base_instructions: str,
    clean_instructions: str,
    clean_signal_text: str = DEFAULT_CLEAN_SIGNAL,
    allow_missing_answer_for_incorrect: bool = False,
) -> CuratedTrace:
    """Curate an SFT example for context-reset training."""

    normalized = normalize_trace(raw_trace)
    instructions = _combine_instructions(base_instructions, clean_instructions)

    if is_correct:
        if normalized.answer_text is None:
            raise ValueError("Correct traces must end with a final <answer> block.")
        response = normalized.as_response()
        return CuratedTrace(
            instructions=instructions,
            response=response,
            uses_clean=False,
            normalized_trace=normalized,
        )

    raw_requests_clean = "<clean>" in raw_trace.lower()
    if (
        normalized.answer_text is None
        and not raw_requests_clean
        and not allow_missing_answer_for_incorrect
    ):
        raise ValueError(
            "Incorrect traces must include a final <answer> block unless they are "
            "explicitly marked as clean-only negatives or already request <clean>."
        )

    think_text = normalized.think_text
    if clean_signal_text.lower() not in think_text.lower():
        think_text = f"{think_text}\n\n{clean_signal_text}"

    response = f"<think>\n{think_text}\n</think>\n<clean>"
    return CuratedTrace(
        instructions=instructions,
        response=response,
        uses_clean=True,
        normalized_trace=normalized,
    )

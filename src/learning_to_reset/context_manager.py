"""One-shot context management for reset-aware generation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional


ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ManagedGeneration:
    """Represents a full one-shot clean interaction."""

    initial_prompt: str
    retry_prompt: Optional[str]
    initial_response: str
    retry_response: Optional[str]
    cleaned: bool
    final_response: str
    final_answer: Optional[str]


def build_initial_prompt(
    question: str,
    base_instructions: str,
    clean_instructions: str,
) -> str:
    parts = [base_instructions.strip(), clean_instructions.strip(), f"Question: {question.strip()}"]
    return "\n\n".join(part for part in parts if part)


def build_retry_prompt(question: str, base_instructions: str) -> str:
    parts = [base_instructions.strip(), f"Question: {question.strip()}"]
    return "\n\n".join(part for part in parts if part)


def extract_answer_text(response: str) -> Optional[str]:
    answers = ANSWER_PATTERN.findall(response)
    if not answers:
        return None
    return answers[-1].strip()


def manage_single_clean_cycle(
    *,
    question: str,
    base_instructions: str,
    clean_instructions: str,
    initial_response: str,
    retry_response: Optional[str] = None,
    clean_token: str = "<clean>",
) -> ManagedGeneration:
    """Execute the current single-use context reset protocol."""

    initial_prompt = build_initial_prompt(question, base_instructions, clean_instructions)
    cleaned = clean_token in initial_response

    if cleaned and retry_response is None:
        raise ValueError("A retry response is required when the initial response emits <clean>.")

    if cleaned:
        retry_prompt = build_retry_prompt(question, base_instructions)
        final_response = retry_response or ""
    else:
        retry_prompt = None
        final_response = initial_response

    final_answer = extract_answer_text(final_response)
    return ManagedGeneration(
        initial_prompt=initial_prompt,
        retry_prompt=retry_prompt,
        initial_response=initial_response,
        retry_response=retry_response,
        cleaned=cleaned,
        final_response=final_response,
        final_answer=final_answer,
    )

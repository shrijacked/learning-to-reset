"""Bounded multi-step cleaning utilities for the extension track."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional, Sequence, Tuple

from learning_to_reset.context_manager import (
    build_initial_prompt,
    build_retry_prompt,
    extract_answer_text,
    response_requests_clean_retry,
)
from learning_to_reset.data import CountdownSample
from learning_to_reset.rollout_runtime import RewardBreakdown, compute_countdown_reward


@dataclass(frozen=True)
class MultiCleanSegment:
    """One generated segment inside a bounded multi-clean interaction."""

    prompt: str
    response: str
    requested_clean: bool


@dataclass(frozen=True)
class ManagedMultiCleanGeneration:
    """A bounded multi-clean interaction over one or more generated segments."""

    segments: Tuple[MultiCleanSegment, ...]
    clean_count: int
    max_cleans: int
    budget_exhausted: bool
    final_response: str
    final_answer: Optional[str]
    retained_notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MultiCleanTrajectory:
    """Reward-attached version of a bounded multi-clean interaction."""

    managed: ManagedMultiCleanGeneration
    segment_rewards: Tuple[RewardBreakdown, ...]
    clean_penalty: float
    adjusted_total_reward: float


def manage_bounded_clean_cycles(
    *,
    question: str,
    base_instructions: str,
    clean_instructions: str,
    responses: Sequence[str],
    max_cleans: int,
    clean_token: str = "<clean>",
) -> ManagedMultiCleanGeneration:
    """Allow repeated cleaning up to a fixed budget, then stop on the last response."""

    if max_cleans < 0:
        raise ValueError("max_cleans must be non-negative.")
    if not responses:
        raise ValueError("At least one response is required.")

    initial_prompt = build_initial_prompt(question, base_instructions, clean_instructions)
    retry_prompt = build_retry_prompt(question, base_instructions)

    segments = []
    clean_count = 0
    for index, response in enumerate(responses):
        requested_clean = response_requests_clean_retry(response, clean_token=clean_token)
        segments.append(
            MultiCleanSegment(
                prompt=initial_prompt if index == 0 else retry_prompt,
                response=response,
                requested_clean=requested_clean,
            )
        )

        if not requested_clean:
            final_answer = extract_answer_text(response)
            return ManagedMultiCleanGeneration(
                segments=tuple(segments),
                clean_count=clean_count,
                max_cleans=max_cleans,
                budget_exhausted=False,
                final_response=response,
                final_answer=final_answer,
            )

        if clean_count >= max_cleans:
            return ManagedMultiCleanGeneration(
                segments=tuple(segments),
                clean_count=clean_count,
                max_cleans=max_cleans,
                budget_exhausted=True,
                final_response=response,
                final_answer=extract_answer_text(response),
            )

        clean_count += 1
        if index == len(responses) - 1:
            raise ValueError("A follow-up response is required while clean budget remains.")

    raise RuntimeError("Bounded clean management finished without a terminal response.")


def extract_retained_notes(
    response: str,
    *,
    retain_start: str = "<retain>",
    retain_end: str = "</retain>",
) -> Tuple[str, ...]:
    """Extract explicit snippets that should survive a clean reset."""

    pattern = re.compile(
        f"{re.escape(retain_start)}(.*?){re.escape(retain_end)}",
        flags=re.DOTALL,
    )
    notes = []
    for match in pattern.finditer(response):
        note = " ".join(match.group(1).strip().split())
        if note:
            notes.append(note)
    return tuple(notes)


def _append_unique_notes(
    existing: Sequence[str],
    new_notes: Sequence[str],
) -> Tuple[str, ...]:
    retained = list(existing)
    seen = set(retained)
    for note in new_notes:
        if note not in seen:
            retained.append(note)
            seen.add(note)
    return tuple(retained)


def build_retained_retry_prompt(
    *,
    question: str,
    base_instructions: str,
    retained_notes: Sequence[str],
) -> str:
    """Build a retry prompt that keeps only explicit retained notes."""

    retry_prompt = build_retry_prompt(question, base_instructions)
    notes = tuple(note.strip() for note in retained_notes if note.strip())
    if not notes:
        return retry_prompt

    retained_block = "\n".join(f"- {note}" for note in notes)
    return (
        f"{retry_prompt}\n\n"
        "Retained notes from previous attempts:\n"
        f"{retained_block}"
    )


def manage_selective_retention_clean_cycles(
    *,
    question: str,
    base_instructions: str,
    clean_instructions: str,
    responses: Sequence[str],
    max_cleans: int,
    clean_token: str = "<clean>",
) -> ManagedMultiCleanGeneration:
    """Allow bounded cleaning while carrying only explicit retained notes forward."""

    if max_cleans < 0:
        raise ValueError("max_cleans must be non-negative.")
    if not responses:
        raise ValueError("At least one response is required.")

    initial_prompt = build_initial_prompt(question, base_instructions, clean_instructions)
    segments = []
    clean_count = 0
    retained_notes: Tuple[str, ...] = ()

    for index, response in enumerate(responses):
        prompt = (
            initial_prompt
            if index == 0
            else build_retained_retry_prompt(
                question=question,
                base_instructions=base_instructions,
                retained_notes=retained_notes,
            )
        )
        requested_clean = response_requests_clean_retry(response, clean_token=clean_token)
        segments.append(
            MultiCleanSegment(
                prompt=prompt,
                response=response,
                requested_clean=requested_clean,
            )
        )

        if not requested_clean:
            final_answer = extract_answer_text(response)
            return ManagedMultiCleanGeneration(
                segments=tuple(segments),
                clean_count=clean_count,
                max_cleans=max_cleans,
                budget_exhausted=False,
                final_response=response,
                final_answer=final_answer,
                retained_notes=retained_notes,
            )

        retained_notes = _append_unique_notes(
            retained_notes,
            extract_retained_notes(response),
        )
        if clean_count >= max_cleans:
            return ManagedMultiCleanGeneration(
                segments=tuple(segments),
                clean_count=clean_count,
                max_cleans=max_cleans,
                budget_exhausted=True,
                final_response=response,
                final_answer=extract_answer_text(response),
                retained_notes=retained_notes,
            )

        clean_count += 1
        if index == len(responses) - 1:
            raise ValueError("A follow-up response is required while clean budget remains.")

    raise RuntimeError("Selective retention management finished without a terminal response.")


def build_multi_clean_trajectory(
    *,
    question: str,
    sample: CountdownSample,
    responses: Sequence[str],
    max_cleans: int,
    clean_step_penalty: float = 0.05,
    base_instructions: str,
    clean_instructions: str,
) -> MultiCleanTrajectory:
    """Compute extension-style reward accounting for bounded multi-clean behavior."""

    managed = manage_bounded_clean_cycles(
        question=question,
        base_instructions=base_instructions,
        clean_instructions=clean_instructions,
        responses=responses,
        max_cleans=max_cleans,
    )
    segment_rewards = tuple(
        compute_countdown_reward(segment.response, sample)
        for segment in managed.segments
    )
    clean_penalty = clean_step_penalty * managed.clean_count
    adjusted_total_reward = segment_rewards[-1].total_reward - clean_penalty
    return MultiCleanTrajectory(
        managed=managed,
        segment_rewards=segment_rewards,
        clean_penalty=clean_penalty,
        adjusted_total_reward=adjusted_total_reward,
    )


def build_selective_retention_trajectory(
    *,
    question: str,
    sample: CountdownSample,
    responses: Sequence[str],
    max_cleans: int,
    clean_step_penalty: float = 0.05,
    base_instructions: str,
    clean_instructions: str,
) -> MultiCleanTrajectory:
    """Compute reward accounting for the selective-retention extension path."""

    managed = manage_selective_retention_clean_cycles(
        question=question,
        base_instructions=base_instructions,
        clean_instructions=clean_instructions,
        responses=responses,
        max_cleans=max_cleans,
    )
    segment_rewards = tuple(
        compute_countdown_reward(segment.response, sample)
        for segment in managed.segments
    )
    clean_penalty = clean_step_penalty * managed.clean_count
    adjusted_total_reward = segment_rewards[-1].total_reward - clean_penalty
    return MultiCleanTrajectory(
        managed=managed,
        segment_rewards=segment_rewards,
        clean_penalty=clean_penalty,
        adjusted_total_reward=adjusted_total_reward,
    )

"""Bounded multi-step cleaning utilities for the extension track."""

from __future__ import annotations

from dataclasses import dataclass
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


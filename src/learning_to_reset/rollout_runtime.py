"""Runtime helpers that connect generated responses to rewards and trajectories."""

from __future__ import annotations

from dataclasses import dataclass

from learning_to_reset.context_manager import manage_single_clean_cycle
from learning_to_reset.countdown_verifier import score_countdown_response
from learning_to_reset.data import CountdownSample
from learning_to_reset.prompts import DEFAULT_BASE_INSTRUCTIONS, DEFAULT_CLEAN_INSTRUCTIONS
from learning_to_reset.rloo import TrajectorySample


@dataclass(frozen=True)
class RewardBreakdown:
    """Formatting and correctness components for one response."""

    total_reward: float
    format_reward: float
    correctness_reward: float
    has_valid_format: bool
    is_correct: bool
    verification_reason: str


@dataclass(frozen=True)
class CleanTrajectory:
    """A one-shot clean interaction with reward information attached."""

    cleaned: bool
    final_response: str
    initial_reward: RewardBreakdown
    retry_reward: RewardBreakdown | None
    final_reward: RewardBreakdown
    trajectory_sample: TrajectorySample


def compute_countdown_reward(
    response: str,
    sample: CountdownSample,
    *,
    formatting_reward: float = 0.1,
    correctness_reward: float = 1.0,
) -> RewardBreakdown:
    """Compute formatting and correctness rewards for a tagged response."""

    has_valid_format = "<answer>" in response and "</answer>" in response
    verification = score_countdown_response(response, sample)
    correct = verification.is_valid and verification.reaches_target

    format_value = formatting_reward if has_valid_format else 0.0
    correct_value = correctness_reward if correct else 0.0
    return RewardBreakdown(
        total_reward=format_value + correct_value,
        format_reward=format_value,
        correctness_reward=correct_value,
        has_valid_format=has_valid_format,
        is_correct=correct,
        verification_reason=verification.reason,
    )


def build_clean_trajectory(
    *,
    question: str,
    sample: CountdownSample,
    initial_response: str,
    retry_response: str | None = None,
    base_instructions: str = DEFAULT_BASE_INSTRUCTIONS,
    clean_instructions: str = DEFAULT_CLEAN_INSTRUCTIONS,
    initial_length: int | None = None,
    retry_length: int | None = None,
    token_length_fn=None,
) -> CleanTrajectory:
    """Assemble a clean-aware trajectory and the matching RLOO sample."""

    managed = manage_single_clean_cycle(
        question=question,
        base_instructions=base_instructions,
        clean_instructions=clean_instructions,
        initial_response=initial_response,
        retry_response=retry_response,
    )

    initial_reward = compute_countdown_reward(initial_response, sample)
    retry_reward = compute_countdown_reward(retry_response, sample) if retry_response else None
    final_reward = retry_reward if retry_reward is not None else initial_reward

    if token_length_fn is None:
        token_length_fn = lambda text: max(1, len(text.split()))
    if initial_length is None:
        initial_length = token_length_fn(initial_response)
    if retry_response is not None and retry_length is None:
        retry_length = token_length_fn(retry_response)

    trajectory_sample = TrajectorySample(
        initial_reward=initial_reward.total_reward,
        initial_length=initial_length,
        retry_reward=(retry_reward.total_reward if retry_reward is not None else None),
        retry_length=retry_length,
    )

    return CleanTrajectory(
        cleaned=managed.cleaned,
        final_response=managed.final_response,
        initial_reward=initial_reward,
        retry_reward=retry_reward,
        final_reward=final_reward,
        trajectory_sample=trajectory_sample,
    )

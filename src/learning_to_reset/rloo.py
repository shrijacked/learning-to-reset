"""Modified RLOO utilities for reset-aware training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class TrajectorySample:
    """One interaction sample containing y0 and optional y1 statistics."""

    initial_reward: float
    initial_length: int
    retry_reward: Optional[float] = None
    retry_length: Optional[int] = None

    def __post_init__(self) -> None:
        if self.initial_length <= 0:
            raise ValueError("Initial length must be positive.")
        if self.retry_reward is None and self.retry_length is not None:
            raise ValueError("Retry length cannot be provided without a retry reward.")
        if self.retry_reward is not None:
            if self.retry_length is None:
                raise ValueError("Retry length is required when retry reward is provided.")
            if self.retry_length <= 0:
                raise ValueError("Retry length must be positive.")

    @property
    def uses_clean(self) -> bool:
        return self.retry_reward is not None


@dataclass(frozen=True)
class TrajectoryTerm:
    """Per-trajectory terms induced by the modified RLOO update."""

    total_reward: float
    advantage: float
    initial_scale: float
    retry_scale: float


@dataclass(frozen=True)
class ModifiedRLOOResult:
    """Batch-level result for the modified RLOO computation."""

    normalization: int
    clean_trajectory_count: int
    terms: Tuple[TrajectoryTerm, ...]


def compute_total_reward(initial_reward: float, retry_reward: Optional[float]) -> float:
    """Return the final rewarded outcome from Eq. 4."""

    return initial_reward if retry_reward is None else retry_reward


def compute_leave_one_out_advantages(rewards: Sequence[float]) -> Tuple[float, ...]:
    """Compute leave-one-out advantages from Eq. 5."""

    count = len(rewards)
    if count < 2:
        raise ValueError("Modified RLOO requires at least two rewards for leave-one-out baselines.")

    total_reward = sum(rewards)
    advantages = []
    for reward in rewards:
        baseline = (total_reward - reward) / (count - 1)
        advantages.append(reward - baseline)
    return tuple(advantages)


def compute_modified_rloo_terms(
    trajectories: Sequence[TrajectorySample],
) -> ModifiedRLOOResult:
    """Translate Eq. 4-6 into explicit per-segment scaling terms."""

    if len(trajectories) < 2:
        raise ValueError("At least two trajectories are required for modified RLOO.")

    interaction_rewards = tuple(
        compute_total_reward(
            initial_reward=trajectory.initial_reward,
            retry_reward=trajectory.retry_reward,
        )
        for trajectory in trajectories
    )
    advantages = compute_leave_one_out_advantages(interaction_rewards)

    clean_trajectory_count = sum(1 for trajectory in trajectories if trajectory.uses_clean)
    normalization = len(trajectories) + clean_trajectory_count

    terms = []
    for trajectory, reward, advantage in zip(trajectories, interaction_rewards, advantages):
        initial_scale = advantage / (normalization * trajectory.initial_length)
        retry_scale = 0.0
        if trajectory.uses_clean and trajectory.retry_length is not None:
            retry_scale = advantage / (normalization * trajectory.retry_length)

        terms.append(
            TrajectoryTerm(
                total_reward=reward,
                advantage=advantage,
                initial_scale=initial_scale,
                retry_scale=retry_scale,
            )
        )

    return ModifiedRLOOResult(
        normalization=normalization,
        clean_trajectory_count=clean_trajectory_count,
        terms=tuple(terms),
    )

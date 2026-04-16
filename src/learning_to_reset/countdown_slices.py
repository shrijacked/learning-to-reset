"""Deterministic Countdown subset helpers for paper-style evaluation slices."""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence, Tuple

from learning_to_reset.countdown_solver import solve_countdown
from learning_to_reset.data import CountdownSample


def expression_uses_multiplication_or_division(expression: str | None) -> bool:
    """Return whether an arithmetic expression uses a multiplication or division operator."""

    if not expression:
        return False
    return "*" in expression or "/" in expression


def _state_key(values: Sequence[Fraction]) -> Tuple[Tuple[int, int], ...]:
    return tuple(sorted((value.numerator, value.denominator) for value in values))


def can_reach_with_add_sub_only(numbers: Sequence[int], target: int) -> bool:
    """Check whether a Countdown target is reachable using only addition/subtraction."""

    target_value = Fraction(int(target))
    initial = tuple(Fraction(int(number)) for number in numbers)
    seen_states = set()

    def search(values: Tuple[Fraction, ...]) -> bool:
        if target_value in values:
            return True
        if len(values) < 2:
            return False

        key = _state_key(values)
        if key in seen_states:
            return False
        seen_states.add(key)

        for left_index in range(len(values)):
            for right_index in range(left_index + 1, len(values)):
                left = values[left_index]
                right = values[right_index]
                remainder = tuple(
                    value
                    for value_index, value in enumerate(values)
                    if value_index not in {left_index, right_index}
                )
                for candidate in (left + right, left - right, right - left):
                    if search(remainder + (candidate,)):
                        return True
        return False

    return search(initial)


def _metadata_marks_hard(sample: CountdownSample) -> bool:
    difficulty = sample.metadata.get("difficulty")
    if isinstance(difficulty, str) and difficulty.strip().lower() == "hard":
        return True

    for key in ("hard", "is_hard", "requires_mul_div", "requires_multiplication_or_division"):
        value = sample.metadata.get(key)
        if isinstance(value, bool) and value:
            return True
    return False


def is_hard_countdown_sample(sample: CountdownSample) -> bool:
    """Identify the hard Countdown slice used for multiplication/division-heavy evals."""

    if _metadata_marks_hard(sample):
        return True
    if sample.solution:
        return expression_uses_multiplication_or_division(sample.solution)
    if can_reach_with_add_sub_only(sample.numbers, sample.target):
        return False

    solution = solve_countdown(sample.numbers, sample.target)
    return expression_uses_multiplication_or_division(solution)


def filter_hard_countdown_samples(samples: Sequence[CountdownSample]) -> Tuple[CountdownSample, ...]:
    """Return Countdown samples from the deterministic hard subset, preserving input order."""

    return tuple(sample for sample in samples if is_hard_countdown_sample(sample))

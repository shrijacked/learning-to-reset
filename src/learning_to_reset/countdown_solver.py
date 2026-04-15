"""Deterministic Countdown solver used for task-aligned fallback traces."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CountdownExpression:
    """One intermediate Countdown expression and its exact arithmetic value."""

    value: Fraction
    expression: str


def _combine_pair(
    left: CountdownExpression,
    right: CountdownExpression,
) -> Tuple[CountdownExpression, ...]:
    """Combine two intermediate expressions with legal arithmetic operators."""

    candidates = [
        CountdownExpression(
            value=left.value + right.value,
            expression=f"({left.expression} + {right.expression})",
        ),
        CountdownExpression(
            value=left.value - right.value,
            expression=f"({left.expression} - {right.expression})",
        ),
        CountdownExpression(
            value=right.value - left.value,
            expression=f"({right.expression} - {left.expression})",
        ),
        CountdownExpression(
            value=left.value * right.value,
            expression=f"({left.expression} * {right.expression})",
        ),
    ]
    if right.value != 0:
        candidates.append(
            CountdownExpression(
                value=left.value / right.value,
                expression=f"({left.expression} / {right.expression})",
            )
        )
    if left.value != 0:
        candidates.append(
            CountdownExpression(
                value=right.value / left.value,
                expression=f"({right.expression} / {left.expression})",
            )
        )

    deduplicated = []
    seen = set()
    for candidate in candidates:
        key = (candidate.value.numerator, candidate.value.denominator, candidate.expression)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(candidate)
    return tuple(deduplicated)


def solve_countdown_variants(
    numbers: Sequence[int],
    target: int,
    *,
    max_solutions: int = 4,
) -> Tuple[str, ...]:
    """Find multiple legal arithmetic expressions that reach the target."""

    if max_solutions <= 0:
        raise ValueError("max_solutions must be positive.")

    target_value = Fraction(int(target))
    initial = tuple(
        CountdownExpression(value=Fraction(int(number)), expression=str(int(number)))
        for number in numbers
    )
    solutions: List[str] = []
    seen_expressions = set()

    def search(items: Sequence[CountdownExpression]) -> None:
        for item in items:
            if item.value == target_value:
                if item.expression not in seen_expressions:
                    seen_expressions.add(item.expression)
                    solutions.append(item.expression)
                    if len(solutions) >= max_solutions:
                        return

        if len(items) < 2 or len(solutions) >= max_solutions:
            return

        for left_index in range(len(items)):
            for right_index in range(left_index + 1, len(items)):
                left = items[left_index]
                right = items[right_index]
                remainder = tuple(
                    item
                    for item_index, item in enumerate(items)
                    if item_index not in {left_index, right_index}
                )
                for candidate in _combine_pair(left, right):
                    search(remainder + (candidate,))
                    if len(solutions) >= max_solutions:
                        return

    search(initial)
    return tuple(solutions)


def solve_countdown(numbers: Sequence[int], target: int) -> Optional[str]:
    """Find one legal arithmetic expression that reaches the target, if one exists."""

    variants = solve_countdown_variants(numbers, target, max_solutions=1)
    return variants[0] if variants else None

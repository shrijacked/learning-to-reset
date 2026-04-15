"""Verification utilities for Countdown-style arithmetic answers."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Sequence, Tuple

from learning_to_reset.context_manager import extract_answer_text
from learning_to_reset.data import CountdownSample


@dataclass(frozen=True)
class VerificationResult:
    """Result of checking a generated arithmetic expression."""

    expression: Optional[str]
    is_valid: bool
    reaches_target: bool
    used_numbers: Tuple[int, ...]
    value: Optional[Fraction]
    reason: str


def extract_answer_expression(response: str) -> Optional[str]:
    """Extract the final answer expression from a tagged response."""

    answer = extract_answer_text(response)
    return answer.strip() if answer is not None else None


def _evaluate_node(node: ast.AST) -> Tuple[Fraction, Tuple[int, ...]]:
    if isinstance(node, ast.BinOp):
        left_value, left_numbers = _evaluate_node(node.left)
        right_value, right_numbers = _evaluate_node(node.right)

        if isinstance(node.op, ast.Add):
            return left_value + right_value, left_numbers + right_numbers
        if isinstance(node.op, ast.Sub):
            return left_value - right_value, left_numbers + right_numbers
        if isinstance(node.op, ast.Mult):
            return left_value * right_value, left_numbers + right_numbers
        if isinstance(node.op, ast.Div):
            if right_value == 0:
                raise ValueError("Division by zero is not allowed.")
            return left_value / right_value, left_numbers + right_numbers
        raise ValueError("Unsupported binary operator.")

    if isinstance(node, ast.UnaryOp):
        value, numbers = _evaluate_node(node.operand)
        if isinstance(node.op, ast.USub):
            return -value, numbers
        if isinstance(node.op, ast.UAdd):
            return value, numbers
        raise ValueError("Unsupported unary operator.")

    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return Fraction(node.value), (int(node.value),)

    raise ValueError("Only integer constants and basic arithmetic are allowed.")


def verify_countdown_expression(
    expression: str,
    *,
    numbers: Sequence[int],
    target: int,
) -> VerificationResult:
    """Check whether an arithmetic expression is legal and reaches the target."""

    try:
        parsed = ast.parse(expression, mode="eval")
        value, used_numbers = _evaluate_node(parsed.body)
    except SyntaxError as exc:
        return VerificationResult(
            expression=expression,
            is_valid=False,
            reaches_target=False,
            used_numbers=(),
            value=None,
            reason=f"Syntax error: {exc.msg}",
        )
    except ValueError as exc:
        return VerificationResult(
            expression=expression,
            is_valid=False,
            reaches_target=False,
            used_numbers=(),
            value=None,
            reason=str(exc),
        )

    allowed = Counter(int(number) for number in numbers)
    used = Counter(int(number) for number in used_numbers)
    if any(used[number] > allowed[number] for number in used):
        return VerificationResult(
            expression=expression,
            is_valid=False,
            reaches_target=False,
            used_numbers=tuple(int(number) for number in used_numbers),
            value=value,
            reason="Expression uses unused or repeated numbers.",
        )

    reaches_target = value == Fraction(target)
    reason = "Valid expression."
    if not reaches_target:
        reason = "Valid expression, but it does not reach the target."

    return VerificationResult(
        expression=expression,
        is_valid=True,
        reaches_target=reaches_target,
        used_numbers=tuple(int(number) for number in used_numbers),
        value=value,
        reason=reason,
    )


def score_countdown_response(response: str, sample: CountdownSample) -> VerificationResult:
    """Verify the final answer expression inside a tagged model response."""

    expression = extract_answer_expression(response)
    if expression is None:
        return VerificationResult(
            expression=None,
            is_valid=False,
            reaches_target=False,
            used_numbers=(),
            value=None,
            reason="No <answer> block found in response.",
        )

    return verify_countdown_expression(
        expression,
        numbers=sample.numbers,
        target=sample.target,
    )

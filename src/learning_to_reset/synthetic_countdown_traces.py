"""Task-aligned synthetic Countdown traces used as a local fallback trace source."""

from __future__ import annotations

import argparse
import ast
import json
import random
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from learning_to_reset.countdown_solver import solve_countdown, solve_countdown_variants
from learning_to_reset.countdown_verifier import verify_countdown_expression
from learning_to_reset.data import CountdownSample, load_countdown_samples
from learning_to_reset.paper_sources import write_jsonl_records


SingleRecoveryStyle = Literal["walkthrough", "verification", "contrastive", "grounded"]
RecoveryStyle = Literal[
    "walkthrough", "verification", "contrastive", "grounded", "both", "all"
]


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _describe_expression_node(node: ast.AST) -> Tuple[str, Fraction, Tuple[str, ...]]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        value = Fraction(int(node.value))
        return str(int(node.value)), value, ()

    if isinstance(node, ast.UnaryOp):
        operand_expression, operand_value, operand_steps = _describe_expression_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand_expression, operand_value, operand_steps
        if isinstance(node.op, ast.USub):
            expression = f"(-{operand_expression})"
            value = -operand_value
            return (
                expression,
                value,
                operand_steps + (f"Compute {expression} = {_format_fraction(value)}.",),
            )
        raise ValueError("Unsupported unary operator in synthetic walkthrough.")

    if isinstance(node, ast.BinOp):
        left_expression, left_value, left_steps = _describe_expression_node(node.left)
        right_expression, right_value, right_steps = _describe_expression_node(node.right)

        if isinstance(node.op, ast.Add):
            operator_symbol = "+"
            value = left_value + right_value
        elif isinstance(node.op, ast.Sub):
            operator_symbol = "-"
            value = left_value - right_value
        elif isinstance(node.op, ast.Mult):
            operator_symbol = "*"
            value = left_value * right_value
        elif isinstance(node.op, ast.Div):
            if right_value == 0:
                raise ValueError("Division by zero is not allowed in synthetic walkthrough.")
            operator_symbol = "/"
            value = left_value / right_value
        else:
            raise ValueError("Unsupported binary operator in synthetic walkthrough.")

        expression = f"({left_expression} {operator_symbol} {right_expression})"
        step = f"Compute {expression} = {_format_fraction(value)}."
        return expression, value, left_steps + right_steps + (step,)

    raise ValueError("Unsupported syntax in synthetic walkthrough expression.")


def build_solution_walkthrough(
    sample: CountdownSample,
    *,
    solution_expression: str,
    after_reset: bool,
) -> str:
    """Render a short arithmetic walkthrough for a solved Countdown expression."""

    parsed = ast.parse(solution_expression, mode="eval")
    _, value, steps = _describe_expression_node(parsed.body)

    intro = (
        "After resetting the scratch work, I rebuild the solution carefully."
        if after_reset
        else (
            "I can reach "
            f"{sample.target} by combining the numbers {list(sample.numbers)} step by step."
        )
    )
    lines = [intro]
    lines.extend(f"Step {index}: {step}" for index, step in enumerate(steps, start=1))
    lines.append(
        "This gives "
        f"{_format_fraction(value)}, which matches the target {sample.target}."
    )
    return "\n".join(lines)


def build_positive_trace_record(
    sample: CountdownSample,
    *,
    solution_expression: str,
) -> Dict[str, Any]:
    """Create a productive tagged trace for one Countdown sample."""

    walkthrough = build_solution_walkthrough(
        sample,
        solution_expression=solution_expression,
        after_reset=False,
    )
    return {
        "source_id": f"{sample.source_id or 'countdown'}:synthetic-positive",
        "problem": sample.question,
        "raw_trace": (
            "<think>\n"
            f"{walkthrough}\n"
            "</think>\n"
            "<answer>\n"
            f"{solution_expression}\n"
            "</answer>"
        ),
        "is_correct": True,
        "metadata": {
            "source": "synthetic_countdown_solver",
            "numbers": list(sample.numbers),
            "target": sample.target,
            "solution_expression": solution_expression,
        },
    }


def build_recovery_response(
    sample: CountdownSample,
    *,
    solution_expression: str,
) -> str:
    """Create a retry-stage response that rebuilds the solution after cleaning."""

    walkthrough = build_solution_walkthrough(
        sample,
        solution_expression=solution_expression,
        after_reset=True,
    )
    return (
        "<think>\n"
        f"{walkthrough}\n"
        "</think>\n"
        "<answer>\n"
        f"{solution_expression}\n"
        "</answer>"
    )


def build_verified_recovery_response(
    sample: CountdownSample,
    *,
    solution_expression: str,
) -> str:
    """Create a concise retry response that explicitly checks the final expression."""

    verification = verify_countdown_expression(
        solution_expression,
        numbers=sample.numbers,
        target=sample.target,
    )
    if not verification.is_valid or not verification.reaches_target:
        raise ValueError(
            f"Recovery expression does not reach target for sample {sample.source_id!r}."
        )

    value = _format_fraction(verification.value or Fraction(sample.target))
    return (
        "<think>\n"
        "After resetting the scratch work, I verify the candidate before answering.\n"
        f"Candidate expression: {solution_expression}.\n"
        f"Verifier check: {solution_expression} = {value}.\n"
        f"The verifier value matches the target {sample.target}, so this expression is safe.\n"
        "</think>\n"
        "<answer>\n"
        f"{solution_expression}\n"
        "</answer>"
    )


def build_contrastive_recovery_response(
    sample: CountdownSample,
    *,
    incorrect_expression: str,
    solution_expression: str,
) -> str:
    """Create a retry response that rejects the failed candidate before answering."""

    incorrect_verification = verify_countdown_expression(
        incorrect_expression,
        numbers=sample.numbers,
        target=sample.target,
    )
    if not incorrect_verification.is_valid:
        raise ValueError(
            f"Contrastive recovery needs a legal failed expression for sample {sample.source_id!r}."
        )
    if incorrect_verification.reaches_target:
        raise ValueError(
            f"Contrastive recovery failed expression already reaches target for sample {sample.source_id!r}."
        )

    solution_verification = verify_countdown_expression(
        solution_expression,
        numbers=sample.numbers,
        target=sample.target,
    )
    if not solution_verification.is_valid or not solution_verification.reaches_target:
        raise ValueError(
            f"Contrastive recovery solution does not reach target for sample {sample.source_id!r}."
        )

    incorrect_value = _format_fraction(
        incorrect_verification.value
        if incorrect_verification.value is not None
        else Fraction(0)
    )
    solution_value = _format_fraction(
        solution_verification.value
        if solution_verification.value is not None
        else Fraction(sample.target)
    )
    return (
        "<think>\n"
        "After resetting the scratch work, I compare the failed candidate with a verified one.\n"
        f"Rejected candidate: {incorrect_expression} = {incorrect_value}, not {sample.target}.\n"
        f"Verified candidate: {solution_expression} = {solution_value}.\n"
        "The verified candidate matches the target, so I use it as the final answer.\n"
        "</think>\n"
        "<answer>\n"
        f"{solution_expression}\n"
        "</answer>"
    )


def _expression_has_binop(expression: str) -> bool:
    parsed = ast.parse(expression, mode="eval")
    return any(isinstance(node, ast.BinOp) for node in ast.walk(parsed))


def _enrich_rejected_expression(
    sample: CountdownSample,
    rejected_expression: str,
) -> str:
    """Promote a degenerate single-number rejected hypothesis to a 2-arg op."""

    if _expression_has_binop(rejected_expression):
        return rejected_expression
    if len(sample.numbers) < 2:
        return rejected_expression

    a = int(sample.numbers[0])
    b = int(sample.numbers[1])
    for op in ("+", "-", "*"):
        candidate = f"({a} {op} {b})"
        verification = verify_countdown_expression(
            candidate,
            numbers=sample.numbers,
            target=sample.target,
        )
        if verification.is_valid and not verification.reaches_target:
            return candidate
    return rejected_expression


def build_grounded_recovery_response(
    sample: CountdownSample,
    *,
    solution_expression: str,
    rejected_expression: Optional[str] = None,
    rng_seed: int = 0,
) -> str:
    """Render an arithmetic-grounded recovery trace.

    The trace contains: per-`BinOp` substep arithmetic for both a rejected
    hypothesis and the accepted solution, an explicit number-budget audit,
    and a final-value reconciliation. Surface phrasing is randomized by
    `rng_seed` so the model cannot memorize a fixed template.
    """

    solution_verification = verify_countdown_expression(
        solution_expression,
        numbers=sample.numbers,
        target=sample.target,
    )
    if not solution_verification.is_valid or not solution_verification.reaches_target:
        raise ValueError(
            f"Grounded recovery solution does not reach target for sample {sample.source_id!r}."
        )

    if rejected_expression is None:
        rejected_expression = build_negative_expression(sample)
    rejected_expression = _enrich_rejected_expression(sample, rejected_expression)

    rejected_verification = verify_countdown_expression(
        rejected_expression,
        numbers=sample.numbers,
        target=sample.target,
    )
    if not rejected_verification.is_valid:
        raise ValueError(
            f"Grounded recovery rejected expression must be legal for sample {sample.source_id!r}."
        )
    if rejected_verification.reaches_target:
        raise ValueError(
            f"Grounded recovery rejected expression must miss target for sample {sample.source_id!r}."
        )

    rng = random.Random(rng_seed)
    intros = (
        "After resetting, I rebuild from numbers",
        "Reset complete. Restarting carefully with numbers",
        "Cleared the scratch work. Working from numbers",
        "Starting over from a clean slate with numbers",
    )
    label_pairs = (("A", "B"), ("1", "2"), ("X", "Y"))
    accept_phrases = ("Accept hypothesis", "Accept candidate", "Accept path")
    reject_phrases = ("Reject hypothesis", "Reject candidate", "Reject path")

    intro = rng.choice(intros)
    rejected_label, accepted_label = rng.choice(label_pairs)
    accept_phrase = rng.choice(accept_phrases)
    reject_phrase = rng.choice(reject_phrases)
    budget_items = [f"used {n} once" for n in sample.numbers]
    rng.shuffle(budget_items)

    _, rejected_value, rejected_steps = _describe_expression_node(
        ast.parse(rejected_expression, mode="eval").body
    )
    _, solution_value, solution_steps = _describe_expression_node(
        ast.parse(solution_expression, mode="eval").body
    )
    rejected_value_str = _format_fraction(rejected_value)
    solution_value_str = _format_fraction(solution_value)

    def _format_steps(steps: Tuple[str, ...]) -> List[str]:
        if not steps:
            return ["  Direct value with no substeps."]
        return [f"  Step {idx}: {step}" for idx, step in enumerate(steps, start=1)]

    lines: List[str] = [
        f"{intro} {list(sample.numbers)}, target {sample.target}.",
        "",
        f"Hypothesis {rejected_label}: try {rejected_expression}.",
    ]
    lines.extend(_format_steps(rejected_steps))
    lines.append(
        f"  Reconcile: {rejected_value_str} != {sample.target}. "
        f"{reject_phrase} {rejected_label}."
    )
    lines.append("")
    lines.append(f"Hypothesis {accepted_label}: try {solution_expression}.")
    lines.extend(_format_steps(solution_steps))
    lines.append(
        f"  Reconcile: {solution_value_str} == {sample.target}. "
        f"{accept_phrase} {accepted_label}."
    )
    lines.append("")
    n = len(sample.numbers)
    lines.append(
        f"Number budget: {', '.join(budget_items)}. {n} of {n} allowed numbers used."
    )
    lines.append(
        f"Final value check: candidate evaluates to {solution_value_str}, "
        f"matches target {sample.target}."
    )

    body = "\n".join(lines)
    return (
        "<think>\n"
        f"{body}\n"
        "</think>\n"
        "<answer>\n"
        f"{solution_expression}\n"
        "</answer>"
    )


def _recovery_styles_for(style: RecoveryStyle) -> Tuple[SingleRecoveryStyle, ...]:
    if style == "walkthrough":
        return ("walkthrough",)
    if style == "verification":
        return ("verification",)
    if style == "contrastive":
        return ("contrastive",)
    if style == "grounded":
        return ("grounded",)
    if style == "both":
        return ("walkthrough", "verification")
    if style == "all":
        return ("walkthrough", "verification", "contrastive", "grounded")
    raise ValueError(
        "recovery_style must be 'walkthrough', 'verification', 'contrastive', "
        "'grounded', 'both', or 'all'."
    )


def _build_recovery_response_for_style(
    sample: CountdownSample,
    *,
    incorrect_expression: str,
    solution_expression: str,
    recovery_style: SingleRecoveryStyle,
) -> str:
    if recovery_style == "walkthrough":
        return build_recovery_response(
            sample,
            solution_expression=solution_expression,
        )
    if recovery_style == "verification":
        return build_verified_recovery_response(
            sample,
            solution_expression=solution_expression,
        )
    if recovery_style == "grounded":
        return build_grounded_recovery_response(
            sample,
            solution_expression=solution_expression,
            rejected_expression=incorrect_expression,
        )
    return build_contrastive_recovery_response(
        sample,
        incorrect_expression=incorrect_expression,
        solution_expression=solution_expression,
    )


def build_negative_expression(sample: CountdownSample) -> str:
    """Pick a simple legal expression that stays off target for reset supervision."""

    candidate_expressions = [str(int(number)) for number in sample.numbers]
    if len(sample.numbers) >= 2:
        left = int(sample.numbers[0])
        right = int(sample.numbers[1])
        candidate_expressions.extend(
            [
                f"({left} + {right})",
                f"({left} - {right})",
                f"({right} - {left})",
                f"({left} * {right})",
            ]
        )
        if right != 0:
            candidate_expressions.append(f"({left} / {right})")
        if left != 0:
            candidate_expressions.append(f"({right} / {left})")

    for expression in candidate_expressions:
        verification = verify_countdown_expression(
            expression,
            numbers=sample.numbers,
            target=sample.target,
        )
        if verification.is_valid and not verification.reaches_target:
            return expression

    raise ValueError(
        f"Could not build a synthetic incorrect expression for sample {sample.source_id!r}."
    )


def build_negative_trace_record(
    sample: CountdownSample,
    *,
    recovery_style: SingleRecoveryStyle = "walkthrough",
) -> Dict[str, Any]:
    """Create an unproductive tagged trace that the curator can turn into `<clean>`."""

    incorrect_expression = build_negative_expression(sample)
    incorrect_verification = verify_countdown_expression(
        incorrect_expression,
        numbers=sample.numbers,
        target=sample.target,
    )
    solution_expression = sample.solution or solve_countdown(sample.numbers, sample.target)
    if solution_expression is None:
        raise ValueError(
            f"Could not build a recovery response for sample {sample.source_id!r}."
        )
    source_suffix = (
        "synthetic-negative"
        if recovery_style == "walkthrough"
        else f"synthetic-negative-{recovery_style}"
    )
    return {
        "source_id": f"{sample.source_id or 'countdown'}:{source_suffix}",
        "problem": sample.question,
        "raw_trace": (
            "<think>\n"
            "I try a straightforward combination first, but it does not actually reach "
            "the target. My tentative expression is "
            f"{incorrect_expression}, which evaluates to "
            f"{_format_fraction(incorrect_verification.value or Fraction(0))} instead of "
            f"{sample.target}.\n"
            "</think>\n"
            "<answer>\n"
            f"{incorrect_expression}\n"
            "</answer>"
        ),
        "recovery_response": _build_recovery_response_for_style(
            sample,
            incorrect_expression=incorrect_expression,
            solution_expression=solution_expression,
            recovery_style=recovery_style,
        ),
        "is_correct": False,
        "metadata": {
            "source": "synthetic_countdown_solver",
            "recovery_style": recovery_style,
            "numbers": list(sample.numbers),
            "target": sample.target,
            "incorrect_expression": incorrect_expression,
            "solution_expression": solution_expression,
        },
    }


def build_synthetic_trace_records(
    samples: Iterable[CountdownSample],
    *,
    include_negative: bool = True,
    solutions_per_sample: int = 1,
    recovery_style: RecoveryStyle = "walkthrough",
) -> Tuple[List[Dict[str, Any]], int]:
    """Build a paired synthetic trace corpus from Countdown samples."""

    if solutions_per_sample <= 0:
        raise ValueError("solutions_per_sample must be positive.")
    recovery_styles = _recovery_styles_for(recovery_style)

    records: List[Dict[str, Any]] = []
    skipped = 0
    for sample in samples:
        if sample.solution is not None:
            solution_expressions = (sample.solution,)
        else:
            solution_expressions = solve_countdown_variants(
                sample.numbers,
                sample.target,
                max_solutions=solutions_per_sample,
            )
        if not solution_expressions:
            skipped += 1
            continue

        for solution_expression in solution_expressions:
            records.append(
                build_positive_trace_record(
                    sample,
                    solution_expression=solution_expression,
                )
            )
            if include_negative:
                for single_recovery_style in recovery_styles:
                    records.append(
                        build_negative_trace_record(
                            CountdownSample(
                                numbers=sample.numbers,
                                target=sample.target,
                                question=sample.question,
                                source_id=sample.source_id,
                                solution=solution_expression,
                                metadata=dict(sample.metadata),
                            ),
                            recovery_style=single_recovery_style,
                        )
                    )
    return records, skipped


def generate_synthetic_trace_corpus(
    *,
    countdown_path: str | Path,
    output_path: str | Path,
    max_samples: int | None = None,
    include_negative: bool = True,
    solutions_per_sample: int = 1,
    recovery_style: RecoveryStyle = "walkthrough",
) -> Dict[str, Any]:
    """Write a synthetic Countdown-aligned trace corpus to JSONL."""

    samples = list(load_countdown_samples(countdown_path))
    if max_samples is not None:
        samples = samples[:max_samples]

    records, skipped = build_synthetic_trace_records(
        samples,
        include_negative=include_negative,
        solutions_per_sample=solutions_per_sample,
        recovery_style=recovery_style,
    )
    write_jsonl_records(records, output_path)

    summary = {
        "countdown_examples": len(samples),
        "records_written": len(records),
        "skipped_unsolved": skipped,
        "include_negative": include_negative,
        "solutions_per_sample": solutions_per_sample,
        "recovery_style": recovery_style,
        "output_path": str(output_path),
    }
    Path(output_path).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Countdown-aligned traces from local source questions."
    )
    parser.add_argument("--countdown", required=True, help="Countdown JSON/JSONL file.")
    parser.add_argument("--output-path", required=True, help="Synthetic trace JSONL path.")
    parser.add_argument("--max-samples", type=int, help="Optional sample cap.")
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Write only productive traces and skip synthetic negative traces.",
    )
    parser.add_argument(
        "--solutions-per-sample",
        type=int,
        default=1,
        help="Maximum number of distinct solved expressions to emit per Countdown sample.",
    )
    parser.add_argument(
        "--recovery-style",
        choices=(
            "walkthrough",
            "verification",
            "contrastive",
            "grounded",
            "both",
            "all",
        ),
        default="walkthrough",
        help="Retry response style for synthetic negative traces.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = generate_synthetic_trace_corpus(
        countdown_path=args.countdown,
        output_path=args.output_path,
        max_samples=args.max_samples,
        include_negative=not args.positive_only,
        solutions_per_sample=args.solutions_per_sample,
        recovery_style=args.recovery_style,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

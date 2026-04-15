"""Task-aligned synthetic Countdown traces used as a local fallback trace source."""

from __future__ import annotations

import argparse
import ast
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from learning_to_reset.countdown_solver import solve_countdown, solve_countdown_variants
from learning_to_reset.countdown_verifier import verify_countdown_expression
from learning_to_reset.data import CountdownSample, load_countdown_samples
from learning_to_reset.paper_sources import write_jsonl_records


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


def build_negative_trace_record(sample: CountdownSample) -> Dict[str, Any]:
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
    return {
        "source_id": f"{sample.source_id or 'countdown'}:synthetic-negative",
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
        "recovery_response": build_recovery_response(
            sample,
            solution_expression=solution_expression,
        ),
        "is_correct": False,
        "metadata": {
            "source": "synthetic_countdown_solver",
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
) -> Tuple[List[Dict[str, Any]], int]:
    """Build a paired synthetic trace corpus from Countdown samples."""

    if solutions_per_sample <= 0:
        raise ValueError("solutions_per_sample must be positive.")

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
                records.append(
                    build_negative_trace_record(
                        CountdownSample(
                            numbers=sample.numbers,
                            target=sample.target,
                            question=sample.question,
                            source_id=sample.source_id,
                            solution=solution_expression,
                            metadata=dict(sample.metadata),
                        )
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
) -> Dict[str, Any]:
    """Write a synthetic Countdown-aligned trace corpus to JSONL."""

    samples = list(load_countdown_samples(countdown_path))
    if max_samples is not None:
        samples = samples[:max_samples]

    records, skipped = build_synthetic_trace_records(
        samples,
        include_negative=include_negative,
        solutions_per_sample=solutions_per_sample,
    )
    write_jsonl_records(records, output_path)

    summary = {
        "countdown_examples": len(samples),
        "records_written": len(records),
        "skipped_unsolved": skipped,
        "include_negative": include_negative,
        "solutions_per_sample": solutions_per_sample,
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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = generate_synthetic_trace_corpus(
        countdown_path=args.countdown,
        output_path=args.output_path,
        max_samples=args.max_samples,
        include_negative=not args.positive_only,
        solutions_per_sample=args.solutions_per_sample,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

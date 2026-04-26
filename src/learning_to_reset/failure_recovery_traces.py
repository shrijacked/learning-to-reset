"""Mine failed Countdown evals into solver-verified recovery traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from learning_to_reset.countdown_solver import solve_countdown_variants
from learning_to_reset.countdown_verifier import verify_countdown_expression
from learning_to_reset.data import CountdownSample
from learning_to_reset.paper_sources import write_jsonl_records
from learning_to_reset.prompts import PromptExample
from learning_to_reset.sft_runtime import load_prepared_examples
from learning_to_reset.synthetic_countdown_traces import (
    build_contrastive_recovery_response,
    build_grounded_recovery_response,
    build_negative_expression,
    build_recovery_response,
    build_verified_recovery_response,
)


SingleRecoveryStyle = Literal["walkthrough", "verification", "contrastive", "grounded"]
RecoveryStyle = Literal[
    "walkthrough", "verification", "contrastive", "grounded", "both", "all"
]


def _read_jsonl_records(path: str | Path) -> Tuple[Dict[str, Any], ...]:
    source = Path(path)
    records = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return tuple(records)


def _sample_from_example(example: PromptExample) -> CountdownSample:
    metadata = example.metadata
    return CountdownSample(
        numbers=tuple(int(number) for number in metadata["numbers"]),
        target=int(metadata["target"]),
        question=str(metadata.get("question", example.prompt)),
        source_id=metadata.get("source_id"),
        solution=metadata.get("solution"),
        metadata=dict(metadata),
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


def _result_response(result: Mapping[str, Any]) -> str:
    for key in ("final_response", "response", "retry_response", "initial_response"):
        value = result.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _result_is_target_correct(result: Mapping[str, Any]) -> bool:
    return bool(result.get("is_valid")) and bool(result.get("reaches_target"))


def _failed_expression_for_result(
    result: Mapping[str, Any],
    sample: CountdownSample,
) -> Tuple[str, str]:
    expression = result.get("expression")
    if expression not in (None, ""):
        candidate = str(expression)
        verification = verify_countdown_expression(
            candidate,
            numbers=sample.numbers,
            target=sample.target,
        )
        if verification.is_valid and not verification.reaches_target:
            return candidate, "eval_result"

    return build_negative_expression(sample), "synthetic_fallback"


def _raw_trace_for_failure(
    result: Mapping[str, Any],
    *,
    failed_expression: str,
) -> str:
    response = _result_response(result)
    lowered_response = response.lower()
    if "<think>" in lowered_response and "</think>" in lowered_response:
        return response

    reason = str(result.get("reason", "The previous attempt did not reach the target."))
    return (
        "<think>\n"
        f"{reason}\n"
        "This path should be reset before answering again.\n"
        "</think>\n"
        "<answer>\n"
        f"{failed_expression}\n"
        "</answer>"
    )


def _build_recovery_response(
    sample: CountdownSample,
    *,
    failed_expression: str,
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
            rejected_expression=failed_expression,
        )
    return build_contrastive_recovery_response(
        sample,
        incorrect_expression=failed_expression,
        solution_expression=solution_expression,
    )


def _solution_expressions(
    sample: CountdownSample,
    *,
    max_solutions: int,
) -> Tuple[str, ...]:
    if max_solutions <= 0:
        raise ValueError("max_solutions must be positive.")

    if sample.solution not in (None, ""):
        solution = str(sample.solution)
        verification = verify_countdown_expression(
            solution,
            numbers=sample.numbers,
            target=sample.target,
        )
        if verification.is_valid and verification.reaches_target:
            return (solution,)

    return solve_countdown_variants(
        sample.numbers,
        sample.target,
        max_solutions=max_solutions,
    )


def build_failure_recovery_trace_records(
    examples: Sequence[PromptExample],
    failure_results: Iterable[Mapping[str, Any]],
    *,
    max_solutions_per_failure: int = 1,
    recovery_style: RecoveryStyle = "contrastive",
    excluded_source_ids: Optional[Iterable[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build solver-verified recovery trace records from failed eval outputs.

    ``excluded_source_ids`` is a contamination guard. If any failure result's
    source_id appears in the set, mining aborts with a ``ValueError`` rather
    than silently leaking the prompt into the SFT corpus.
    """

    examples_by_source_id = {
        str(example.metadata.get("source_id")): example
        for example in examples
        if example.metadata.get("source_id") not in (None, "")
    }
    recovery_styles = _recovery_styles_for(recovery_style)

    excluded: FrozenSet[str] = frozenset(
        str(source_id) for source_id in (excluded_source_ids or ())
    )
    failure_results_list = list(failure_results)
    if excluded:
        overlap = sorted(
            {
                str(result.get("source_id"))
                for result in failure_results_list
                if result.get("source_id") not in (None, "")
                and str(result.get("source_id")) in excluded
            }
        )
        if overlap:
            preview = ", ".join(overlap[:5])
            suffix = "" if len(overlap) <= 5 else f" (and {len(overlap) - 5} more)"
            raise ValueError(
                "Failure mining contamination detected: "
                f"{len(overlap)} eval source_ids overlap the exclusion set: "
                f"{preview}{suffix}. Mining aborted to keep the holdout clean."
            )

    records: List[Dict[str, Any]] = []
    failures_seen = 0
    skipped_solved_or_valid = 0
    skipped_missing_example = 0
    skipped_unsolved = 0

    for result in failure_results_list:
        source_id = result.get("source_id")
        if source_id in (None, ""):
            skipped_missing_example += 1
            continue

        example = examples_by_source_id.get(str(source_id))
        if example is None:
            skipped_missing_example += 1
            continue

        if _result_is_target_correct(result):
            skipped_solved_or_valid += 1
            continue

        failures_seen += 1
        sample = _sample_from_example(example)
        solutions = _solution_expressions(
            sample,
            max_solutions=max_solutions_per_failure,
        )
        if not solutions:
            skipped_unsolved += 1
            continue

        failed_expression, failed_expression_source = _failed_expression_for_result(
            result,
            sample,
        )
        raw_trace = _raw_trace_for_failure(
            result,
            failed_expression=failed_expression,
        )

        for solution_index, solution_expression in enumerate(solutions, start=1):
            for single_style in recovery_styles:
                records.append(
                    {
                        "source_id": (
                            f"{sample.source_id or 'countdown'}:"
                            f"mined-recovery-{solution_index}-{single_style}"
                        ),
                        "problem": sample.question,
                        "raw_trace": raw_trace,
                        "recovery_response": _build_recovery_response(
                            sample,
                            failed_expression=failed_expression,
                            solution_expression=solution_expression,
                            recovery_style=single_style,
                        ),
                        "is_correct": False,
                        "metadata": {
                            "source": "failure_mined_countdown_solver",
                            "source_eval_id": source_id,
                            "recovery_style": single_style,
                            "numbers": list(sample.numbers),
                            "target": sample.target,
                            "failed_expression": failed_expression,
                            "failed_expression_source": failed_expression_source,
                            "solution_expression": solution_expression,
                            "failure_reason": result.get("reason"),
                        },
                    }
                )

    summary = {
        "eval_results": failures_seen
        + skipped_solved_or_valid
        + skipped_missing_example,
        "failures_seen": failures_seen,
        "records_written": len(records),
        "skipped_solved_or_valid": skipped_solved_or_valid,
        "skipped_missing_example": skipped_missing_example,
        "skipped_unsolved": skipped_unsolved,
        "max_solutions_per_failure": max_solutions_per_failure,
        "recovery_style": recovery_style,
        "excluded_source_ids": len(excluded),
    }
    return records, summary


def _load_source_ids_from_jsonl(path: str | Path) -> Set[str]:
    """Read a prepared/eval JSONL and return the set of source_ids it contains."""

    source = Path(path)
    if not source.exists():
        raise ValueError(f"Exclusion file does not exist: {source}")

    found: Set[str] = set()
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Could not parse JSONL line in exclusion file {source}: {exc}"
            ) from exc
        candidate = record.get("source_id")
        if candidate in (None, ""):
            metadata = record.get("metadata") or {}
            candidate = metadata.get("source_id")
        if candidate not in (None, ""):
            found.add(str(candidate))
    return found


def generate_failure_recovery_trace_corpus(
    *,
    prepared_countdown_path: str | Path,
    eval_results_path: str | Path,
    output_path: str | Path,
    max_solutions_per_failure: int = 1,
    recovery_style: RecoveryStyle = "contrastive",
    exclude_source_id_paths: Optional[Sequence[str | Path]] = None,
) -> Dict[str, Any]:
    """Write mined recovery traces from a prepared Countdown file and eval results.

    ``exclude_source_id_paths`` lists prepared JSONL files whose ``source_id``
    fields must NOT appear in the eval results being mined; if any do, mining
    aborts with ``ValueError``. This is the contamination guard for keeping
    held-out hard slices clean.
    """

    examples = load_prepared_examples(prepared_countdown_path)
    results = _read_jsonl_records(eval_results_path)

    excluded_source_ids: Set[str] = set()
    for excl_path in exclude_source_id_paths or ():
        excluded_source_ids.update(_load_source_ids_from_jsonl(excl_path))

    records, summary = build_failure_recovery_trace_records(
        examples,
        results,
        max_solutions_per_failure=max_solutions_per_failure,
        recovery_style=recovery_style,
        excluded_source_ids=excluded_source_ids,
    )
    write_jsonl_records(records, output_path)
    summary["output_path"] = str(output_path)
    summary["exclude_source_id_paths"] = [
        str(path) for path in (exclude_source_id_paths or ())
    ]
    Path(output_path).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mine failed Countdown eval outputs into verified recovery traces."
    )
    parser.add_argument(
        "--prepared-countdown",
        required=True,
        help="Prepared Countdown JSONL used by the eval run.",
    )
    parser.add_argument(
        "--eval-results",
        required=True,
        help="Evaluation results JSONL to mine for failed examples.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Destination JSONL path for mined recovery traces.",
    )
    parser.add_argument(
        "--max-solutions-per-failure",
        type=int,
        default=1,
        help="Maximum solver variants to emit for each failed prompt.",
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
        default="contrastive",
        help="Recovery response style to generate.",
    )
    parser.add_argument(
        "--exclude-source-ids",
        action="append",
        default=[],
        help=(
            "Path to a prepared Countdown JSONL whose source_ids must NOT "
            "appear in the eval results being mined. May be passed multiple "
            "times. Aborts mining on overlap to keep holdout slices clean."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = generate_failure_recovery_trace_corpus(
        prepared_countdown_path=args.prepared_countdown,
        eval_results_path=args.eval_results,
        output_path=args.output_path,
        max_solutions_per_failure=args.max_solutions_per_failure,
        recovery_style=args.recovery_style,
        exclude_source_id_paths=args.exclude_source_ids,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

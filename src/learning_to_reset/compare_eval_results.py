"""Utilities for comparing two Countdown evaluation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


COMPARISON_METRICS = (
    "accuracy",
    "valid_rate",
    "average_score",
    "clean_rate",
)


def load_evaluation_summary(eval_dir: str | Path) -> Dict[str, Any]:
    """Load the summary emitted by the evaluation runtime."""

    summary_path = Path(eval_dir) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing evaluation summary: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation summary must be a JSON object: {summary_path}")
    return payload


def _numeric_value(summary: Dict[str, Any], key: str) -> Optional[float]:
    value = summary.get(key)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def build_evaluation_comparison(
    *,
    baseline_summary: Dict[str, Any],
    candidate_summary: Dict[str, Any],
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
) -> Dict[str, Any]:
    """Compare two evaluation summaries with candidate-minus-baseline deltas."""

    delta: Dict[str, Optional[float]] = {}
    for metric in COMPARISON_METRICS:
        baseline_value = _numeric_value(baseline_summary, metric)
        candidate_value = _numeric_value(candidate_summary, metric)
        delta[metric] = (
            candidate_value - baseline_value
            if baseline_value is not None and candidate_value is not None
            else None
        )

    notes = []
    baseline_total = baseline_summary.get("total_examples")
    candidate_total = candidate_summary.get("total_examples")
    if baseline_total != candidate_total:
        notes.append(
            "Evaluation totals differ; compare rates rather than raw counts."
        )

    winner = choose_comparison_winner(
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
    )
    return {
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "delta": delta,
        "winner": winner,
        "notes": notes,
    }


def choose_comparison_winner(
    *,
    baseline_summary: Dict[str, Any],
    candidate_summary: Dict[str, Any],
    baseline_label: str,
    candidate_label: str,
) -> str:
    """Choose a winner by correctness first, then validity, then average score."""

    ordering_metrics = ("accuracy", "valid_rate", "average_score")
    for metric in ordering_metrics:
        baseline_value = _numeric_value(baseline_summary, metric)
        candidate_value = _numeric_value(candidate_summary, metric)
        if baseline_value is None or candidate_value is None:
            continue
        if candidate_value > baseline_value:
            return candidate_label
        if candidate_value < baseline_value:
            return baseline_label
    return "tie"


def render_comparison_markdown(comparison: Dict[str, Any]) -> str:
    """Render a compact Markdown comparison table."""

    baseline_label = comparison["baseline_label"]
    candidate_label = comparison["candidate_label"]
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    delta = comparison["delta"]

    lines = [
        "# Evaluation Comparison",
        "",
        f"Winner: `{comparison['winner']}`",
        "",
        "| Metric | " + baseline_label + " | " + candidate_label + " | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in ("total_examples", "correct", "valid", *COMPARISON_METRICS):
        baseline_value = baseline.get(metric, "")
        candidate_value = candidate.get(metric, "")
        delta_value = delta.get(metric, "") if metric in delta else ""
        lines.append(
            f"| `{metric}` | {baseline_value} | {candidate_value} | {delta_value} |"
        )

    notes = comparison.get("notes") or []
    if notes:
        lines.extend(["", "## Notes"])
        lines.extend(f"- {note}" for note in notes)

    return "\n".join(lines) + "\n"


def write_comparison_outputs(comparison: Dict[str, Any], output_dir: str | Path) -> None:
    """Write comparison JSON and Markdown artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (destination / "comparison.md").write_text(
        render_comparison_markdown(comparison),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two evaluation output directories.")
    parser.add_argument("--baseline-dir", required=True, help="Directory containing summary.json.")
    parser.add_argument("--candidate-dir", required=True, help="Directory containing summary.json.")
    parser.add_argument("--output-dir", required=True, help="Directory for comparison outputs.")
    parser.add_argument("--baseline-label", default="raw", help="Label for the baseline run.")
    parser.add_argument(
        "--candidate-label",
        default="reset-aware",
        help="Label for the candidate run.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    comparison = build_evaluation_comparison(
        baseline_summary=load_evaluation_summary(args.baseline_dir),
        candidate_summary=load_evaluation_summary(args.candidate_dir),
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )
    write_comparison_outputs(comparison, args.output_dir)
    print(json.dumps(comparison, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

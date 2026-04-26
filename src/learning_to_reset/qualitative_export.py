"""Export qualitative best/worst samples for Figure 7-style write-ups.

Reads a per-example results JSONL produced by `eval_runtime` (or the
reset-aware/multi-clean variants) and writes a Markdown report containing
the N best (target-correct, used a reset) and N worst (invalid, never
cleaned) cases, sorted by score. The output is a stable, human-readable
artifact that can be cited directly in status reports and papers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def _load_records(path: str | Path) -> List[Dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8")
    records: List[Dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def _is_best_case(record: Mapping[str, Any]) -> bool:
    return (
        bool(record.get("is_valid"))
        and bool(record.get("reaches_target"))
        and bool(record.get("cleaned"))
    )


def _is_worst_case(record: Mapping[str, Any]) -> bool:
    return (
        not bool(record.get("is_valid"))
        and not bool(record.get("cleaned"))
    )


def _score_of(record: Mapping[str, Any]) -> float:
    value = record.get("score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def select_qualitative_samples(
    records: Iterable[Mapping[str, Any]],
    *,
    n_best: int,
    n_worst: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Pick the top-N best and top-N worst records for qualitative reporting."""

    if n_best < 0 or n_worst < 0:
        raise ValueError("n_best and n_worst must be non-negative.")

    record_list = [dict(record) for record in records]
    best = sorted(
        (record for record in record_list if _is_best_case(record)),
        key=lambda r: (-_score_of(r), str(r.get("source_id") or "")),
    )[:n_best]
    worst = sorted(
        (record for record in record_list if _is_worst_case(record)),
        key=lambda r: (_score_of(r), str(r.get("source_id") or "")),
    )[:n_worst]
    return {"best": best, "worst": worst}


def _format_record(record: Mapping[str, Any]) -> str:
    lines = [
        f"### {record.get('source_id', 'unknown')}",
        "",
        f"- score: {record.get('score')}",
        f"- is_valid: {record.get('is_valid')}",
        f"- reaches_target: {record.get('reaches_target')}",
        f"- cleaned: {record.get('cleaned')}",
        f"- expression: `{record.get('expression')}`",
        f"- value: `{record.get('value')}`",
    ]
    reason = record.get("reason")
    if reason:
        lines.append(f"- reason: {reason}")
    response = record.get("response", "")
    lines.extend(["", "```text", str(response), "```", ""])
    return "\n".join(lines)


def render_qualitative_markdown(
    selection: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    n_best: int,
    n_worst: int,
) -> str:
    """Render the selected best/worst samples as a Markdown report."""

    parts: List[str] = ["# Qualitative samples", ""]

    parts.append(f"## Best (target-correct, used reset) [requested {n_best}]")
    parts.append("")
    best_rows = list(selection.get("best", ()))
    if not best_rows:
        parts.append("_No qualifying records._")
        parts.append("")
    else:
        for record in best_rows:
            parts.append(_format_record(record))

    parts.append(f"## Worst (invalid, no reset) [requested {n_worst}]")
    parts.append("")
    worst_rows = list(selection.get("worst", ()))
    if not worst_rows:
        parts.append("_No qualifying records._")
        parts.append("")
    else:
        for record in worst_rows:
            parts.append(_format_record(record))

    return "\n".join(parts).rstrip() + "\n"


def write_qualitative_export(
    *,
    results_path: str | Path,
    output_path: str | Path,
    n_best: int,
    n_worst: int,
) -> Dict[str, Any]:
    """Read a results JSONL and write a Figure 7-style qualitative report."""

    records = _load_records(results_path)
    selection = select_qualitative_samples(records, n_best=n_best, n_worst=n_worst)
    markdown = render_qualitative_markdown(selection, n_best=n_best, n_worst=n_worst)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markdown, encoding="utf-8")
    return {
        "results_path": str(results_path),
        "output_path": str(destination),
        "best_count": len(selection["best"]),
        "worst_count": len(selection["worst"]),
        "total_records": len(records),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Figure 7-aligned qualitative samples from eval results."
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to a results.jsonl produced by eval_runtime.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the qualitative.md report to write.",
    )
    parser.add_argument(
        "--n-best",
        type=int,
        default=5,
        help="Number of best (target-correct, cleaned) samples to include.",
    )
    parser.add_argument(
        "--n-worst",
        type=int,
        default=5,
        help="Number of worst (invalid, never cleaned) samples to include.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = write_qualitative_export(
        results_path=args.results,
        output_path=args.output,
        n_best=args.n_best,
        n_worst=args.n_worst,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

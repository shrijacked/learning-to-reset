"""Runner that scores all three extension controllers on a multi-clean eval.

The B3 wrapper does not call the model itself. It reads the segment-by-segment
responses already produced by ``run_multi_clean_eval_loop`` (stored in the
multi-clean eval's ``results.jsonl``) and feeds the same response sequence to
the three controllers in
``learning_to_reset.extension_comparison.compare_extension_trajectories``:

* ``full_reset`` — the baseline ``manage_bounded_clean_cycles`` policy.
* ``selective_retention`` — keeps only explicit retained notes across
  ``<clean>`` boundaries.
* ``memory`` — recalls the top-N salient memory entries written so far.

Because all three strategies score the same response strings, this is a
controller comparison rather than a strategy A/B that re-rolls the model. The
script docs in ``scripts/run_extension_comparison.py`` make that explicit.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from learning_to_reset.data import CountdownSample
from learning_to_reset.extension_comparison import (
    ExtensionComparison,
    compare_extension_trajectories,
    render_extension_comparison_markdown,
    write_extension_comparison_outputs,
)
from learning_to_reset.prompts import (
    DEFAULT_BASE_INSTRUCTIONS,
    DEFAULT_CLEAN_INSTRUCTIONS,
    parse_reasoning_prompt,
)


@dataclass(frozen=True)
class ExampleWithSegments:
    """Pair a prepared example sample with its multi-clean segment responses."""

    sample: CountdownSample
    base_instructions: str
    clean_instructions: str
    responses: Tuple[str, ...]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _sample_from_prepared(record: Dict[str, Any]) -> CountdownSample:
    metadata = record.get("metadata") or {}
    numbers = tuple(int(n) for n in metadata.get("numbers", ()))
    target = int(metadata.get("target"))
    question = str(metadata.get("question", record.get("prompt", "")))
    return CountdownSample(
        numbers=numbers,
        target=target,
        question=question,
        source_id=metadata.get("source_id"),
        solution=metadata.get("solution"),
        metadata=dict(metadata),
    )


def _instructions_for_prompt(prompt: str) -> Tuple[str, str]:
    """Recover (base, clean) instructions from a prepared prompt."""

    try:
        parts = parse_reasoning_prompt(prompt)
    except ValueError:
        return DEFAULT_BASE_INSTRUCTIONS, DEFAULT_CLEAN_INSTRUCTIONS
    base = parts.base_instructions or DEFAULT_BASE_INSTRUCTIONS
    clean = parts.clean_instructions or DEFAULT_CLEAN_INSTRUCTIONS
    return base, clean


def iter_examples_with_segments(
    *,
    prepared_path: str | Path,
    results_path: str | Path,
) -> Iterator[ExampleWithSegments]:
    """Pair each prepared example with the segment responses produced for it."""

    prepared_records = _read_jsonl(Path(prepared_path))
    result_records = _read_jsonl(Path(results_path))

    segments_by_id: Dict[str, List[str]] = {}
    for record in result_records:
        source_id = record.get("source_id")
        if source_id is None:
            continue
        raw_segments = record.get("segments") or []
        responses = [str(seg.get("response", "")) for seg in raw_segments]
        if not responses and "response" in record:
            responses = [str(record["response"])]
        segments_by_id[str(source_id)] = responses

    for prepared in prepared_records:
        sample = _sample_from_prepared(prepared)
        if sample.source_id is None:
            continue
        responses = segments_by_id.get(str(sample.source_id))
        if not responses:
            continue
        base, clean = _instructions_for_prompt(prepared.get("prompt", ""))
        yield ExampleWithSegments(
            sample=sample,
            base_instructions=base,
            clean_instructions=clean,
            responses=tuple(responses),
        )


def aggregate_extension_comparisons(
    comparisons: Iterable[ExtensionComparison],
) -> Dict[str, Any]:
    """Aggregate per-strategy winners and mean adjusted reward."""

    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    winners: Dict[str, int] = {}
    n = 0
    for comparison in comparisons:
        n += 1
        winners[comparison.winner] = winners.get(comparison.winner, 0) + 1
        for row in comparison.rows:
            totals[row.mode] = totals.get(row.mode, 0.0) + row.adjusted_total_reward
            counts[row.mode] = counts.get(row.mode, 0) + 1

    mean = {
        mode: (totals[mode] / counts[mode]) if counts.get(mode) else 0.0
        for mode in totals
    }
    return {
        "total_examples": n,
        "winners": winners,
        "mean_adjusted_reward": mean,
    }


def run_extension_comparison_on_files(
    *,
    prepared_path: str | Path,
    results_path: str | Path,
    output_dir: str | Path,
    max_cleans: int,
    clean_step_penalty: float = 0.05,
    memory_recall_limit: int = 3,
) -> Dict[str, Any]:
    """End-to-end driver: read inputs, score, write outputs, return summary."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    comparisons: List[ExtensionComparison] = []
    per_example: List[Dict[str, Any]] = []
    skipped_incomplete_clean_segments = 0

    for paired in iter_examples_with_segments(
        prepared_path=prepared_path,
        results_path=results_path,
    ):
        responses = list(paired.responses)
        if not responses:
            continue
        try:
            comparison = compare_extension_trajectories(
                sample=paired.sample,
                base_instructions=paired.base_instructions,
                clean_instructions=paired.clean_instructions,
                max_cleans=max_cleans,
                full_reset_responses=responses,
                selective_retention_responses=responses,
                memory_responses=responses,
                clean_step_penalty=clean_step_penalty,
                memory_recall_limit=memory_recall_limit,
            )
        except ValueError as exc:
            if "follow-up response is required" not in str(exc):
                raise
            skipped_incomplete_clean_segments += 1
            continue
        comparisons.append(comparison)
        per_example.append(
            {
                "source_id": paired.sample.source_id,
                "winner": comparison.winner,
                "rows": [asdict(row) for row in comparison.rows],
            }
        )

    summary = aggregate_extension_comparisons(comparisons)

    # Persist a flat JSON we can diff and a Markdown that mirrors Figure 6's
    # comparison style.
    summary_with_examples = dict(summary)
    summary_with_examples[
        "skipped_incomplete_clean_segments"
    ] = skipped_incomplete_clean_segments
    summary_with_examples["per_example"] = per_example
    (output_root / "extension-comparison.json").write_text(
        json.dumps(summary_with_examples, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    md_lines: List[str] = [
        "# Extension Comparison",
        "",
        f"Total examples: **{summary['total_examples']}**",
        f"Skipped incomplete clean-segment outputs: **{skipped_incomplete_clean_segments}**",
        "",
        "## Winners",
        "",
        "| Mode | Wins |",
        "| --- | ---: |",
    ]
    for mode in sorted(summary["winners"]):
        md_lines.append(f"| {mode} | {summary['winners'][mode]} |")
    md_lines += [
        "",
        "## Mean adjusted reward",
        "",
        "| Mode | Mean reward |",
        "| --- | ---: |",
    ]
    for mode in sorted(summary["mean_adjusted_reward"]):
        md_lines.append(
            f"| {mode} | {summary['mean_adjusted_reward'][mode]:.4f} |"
        )
    (output_root / "extension-comparison.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )
    return summary_with_examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score the full-reset, selective-retention, and memory "
            "controllers on the same multi-clean eval segments."
        )
    )
    parser.add_argument(
        "--prepared-countdown",
        required=True,
        help="Prepared Countdown JSONL file (matches eval input).",
    )
    parser.add_argument(
        "--results-jsonl",
        required=True,
        help="results.jsonl produced by run_multi_clean_eval_loop.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for extension-comparison.{json,md}.",
    )
    parser.add_argument(
        "--max-cleans",
        type=int,
        default=3,
        help="Clean budget passed to each controller.",
    )
    parser.add_argument(
        "--clean-step-penalty",
        type=float,
        default=0.05,
        help="Per-clean reward penalty (matches paper Section 3.4).",
    )
    parser.add_argument(
        "--memory-recall-limit",
        type=int,
        default=3,
        help="Top-N memory entries the memory controller recalls.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_extension_comparison_on_files(
        prepared_path=args.prepared_countdown,
        results_path=args.results_jsonl,
        output_dir=args.output_dir,
        max_cleans=args.max_cleans,
        clean_step_penalty=args.clean_step_penalty,
        memory_recall_limit=args.memory_recall_limit,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

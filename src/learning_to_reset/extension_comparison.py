"""Compare extension controllers on the same Countdown interaction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from learning_to_reset.data import CountdownSample
from learning_to_reset.memory_extension import build_memory_clean_trajectory
from learning_to_reset.multi_clean_extension import (
    build_multi_clean_trajectory,
    build_selective_retention_trajectory,
)


@dataclass(frozen=True)
class ExtensionComparisonRow:
    """Compact score row for one extension controller."""

    mode: str
    clean_count: int
    budget_exhausted: bool
    final_answer: str | None
    total_reward: float
    adjusted_total_reward: float


@dataclass(frozen=True)
class ExtensionComparison:
    """Side-by-side result for full reset, selective retention, and memory."""

    rows: tuple[ExtensionComparisonRow, ...]
    winner: str


def _row_from_trajectory(mode: str, trajectory: Any) -> ExtensionComparisonRow:
    return ExtensionComparisonRow(
        mode=mode,
        clean_count=trajectory.managed.clean_count,
        budget_exhausted=trajectory.managed.budget_exhausted,
        final_answer=trajectory.managed.final_answer,
        total_reward=trajectory.segment_rewards[-1].total_reward,
        adjusted_total_reward=trajectory.adjusted_total_reward,
    )


def compare_extension_trajectories(
    *,
    sample: CountdownSample,
    base_instructions: str,
    clean_instructions: str,
    max_cleans: int,
    full_reset_responses: Sequence[str],
    selective_retention_responses: Sequence[str],
    memory_responses: Sequence[str],
    clean_step_penalty: float = 0.05,
    memory_recall_limit: int = 3,
) -> ExtensionComparison:
    """Score the three extension controllers on prepared response sequences."""

    rows = (
        _row_from_trajectory(
            "full_reset",
            build_multi_clean_trajectory(
                question=sample.question,
                sample=sample,
                responses=full_reset_responses,
                max_cleans=max_cleans,
                clean_step_penalty=clean_step_penalty,
                base_instructions=base_instructions,
                clean_instructions=clean_instructions,
            ),
        ),
        _row_from_trajectory(
            "selective_retention",
            build_selective_retention_trajectory(
                question=sample.question,
                sample=sample,
                responses=selective_retention_responses,
                max_cleans=max_cleans,
                clean_step_penalty=clean_step_penalty,
                base_instructions=base_instructions,
                clean_instructions=clean_instructions,
            ),
        ),
        _row_from_trajectory(
            "memory",
            build_memory_clean_trajectory(
                question=sample.question,
                sample=sample,
                responses=memory_responses,
                max_cleans=max_cleans,
                clean_step_penalty=clean_step_penalty,
                base_instructions=base_instructions,
                clean_instructions=clean_instructions,
                recall_limit=memory_recall_limit,
            ),
        ),
    )
    winner = max(rows, key=lambda row: row.adjusted_total_reward).mode
    return ExtensionComparison(rows=rows, winner=winner)


def render_extension_comparison_markdown(comparison: ExtensionComparison) -> str:
    """Render a compact Markdown table for an extension comparison."""

    lines = [
        "# Extension Comparison",
        "",
        f"Winner: `{comparison.winner}`",
        "",
        "| Mode | Clean count | Final answer | Adjusted reward |",
        "| --- | ---: | --- | ---: |",
    ]
    for row in comparison.rows:
        final_answer = row.final_answer or ""
        lines.append(
            f"| {row.mode} | {row.clean_count} | `{final_answer}` | "
            f"{row.adjusted_total_reward:.3f} |"
        )
    return "\n".join(lines) + "\n"


def write_extension_comparison_outputs(
    comparison: ExtensionComparison,
    *,
    output_dir: str | Path,
) -> Dict[str, Path]:
    """Write JSON and Markdown artifacts for an extension comparison."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "extension-comparison.json"
    markdown_path = output_root / "extension-comparison.md"
    payload = {
        "winner": comparison.winner,
        "rows": [asdict(row) for row in comparison.rows],
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_extension_comparison_markdown(comparison),
        encoding="utf-8",
    )
    return {"json": json_path, "markdown": markdown_path}

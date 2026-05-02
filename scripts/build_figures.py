#!/usr/bin/env python
"""Build publication-facing SVG figures from local eval artifacts.

The script prefers checked local summaries under ``tmp/paper-eval`` and falls
back to the documented pilot metrics when those artifacts are absent. It uses
only the Python standard library so the README figures can be regenerated on a
fresh clone without installing plotting packages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]


FALLBACK_RUNS: Dict[str, Dict[str, Any]] = {
    "raw_0_5b": {
        "label": "raw 0.5B",
        "total_examples": 32,
        "valid": 0,
        "correct": 0,
        "valid_rate": 0.0,
        "accuracy": 0.0,
        "clean_rate": 0.0,
        "score_when_cleaned": 0.0,
    },
    "reset_0_5b": {
        "label": "reset-aware 0.5B",
        "total_examples": 32,
        "valid": 29,
        "correct": 1,
        "valid_rate": 29 / 32,
        "accuracy": 1 / 32,
        "clean_rate": 1.0,
        "score_when_cleaned": 0.13125,
    },
    "grounded_fresh32": {
        "label": "grounded fresh-32",
        "total_examples": 32,
        "valid": 31,
        "correct": 0,
        "valid_rate": 31 / 32,
        "accuracy": 0.0,
        "clean_rate": 1.0,
        "score_when_cleaned": 0.096875,
    },
    "grounded_seed131": {
        "label": "grounded seed=131",
        "total_examples": 32,
        "valid": 32,
        "correct": 0,
        "valid_rate": 1.0,
        "accuracy": 0.0,
        "clean_rate": 1.0,
        "score_when_cleaned": 0.1,
    },
    "paper_1b": {
        "label": "paper 1B",
        "total_examples": 100,
        "valid": None,
        "correct": 36.94,
        "valid_rate": None,
        "accuracy": 0.3694,
        "clean_rate": None,
        "score_when_cleaned": None,
    },
}


SUMMARY_PATHS = {
    "raw_0_5b": REPO_ROOT
    / "tmp/paper-eval/sft-hard-focus-plus-mined-fresh-eval-e0.5-hard-raw/summary.json",
    "reset_0_5b": REPO_ROOT
    / "tmp/paper-eval/sft-hard-focus-plus-mined-fresh-eval-e0.5-hard/summary.json",
    "grounded_fresh32": REPO_ROOT
    / "tmp/paper-eval/sft-grounded-26apr-seed219mine-fresh32-multiclean3/summary.json",
    "grounded_seed131": REPO_ROOT
    / "tmp/paper-eval/sft-grounded-26apr-seed219mine-seed131-multiclean3/summary.json",
}


RESULTS_PATHS = {
    "raw_0_5b": REPO_ROOT
    / "tmp/paper-eval/sft-hard-focus-plus-mined-fresh-eval-e0.5-hard-raw/results.jsonl",
    "reset_0_5b": REPO_ROOT
    / "tmp/paper-eval/sft-hard-focus-plus-mined-fresh-eval-e0.5-hard/results.jsonl",
}


def _read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_results(path: Path) -> Dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(rows)
    valid = sum(1 for row in rows if row.get("is_valid"))
    correct = sum(1 for row in rows if row.get("is_valid") and row.get("reaches_target"))
    cleaned = sum(1 for row in rows if row.get("cleaned") or "<clean>" in str(row.get("response", "")).lower())
    cleaned_score = sum(float(row.get("score", 0.0)) for row in rows if row.get("cleaned"))
    return {
        "total_examples": total,
        "valid": valid,
        "correct": correct,
        "valid_rate": valid / total if total else 0.0,
        "accuracy": correct / total if total else 0.0,
        "clean_rate": cleaned / total if total else 0.0,
        "score_when_cleaned": cleaned_score / cleaned if cleaned else 0.0,
    }


def load_metrics() -> Dict[str, Dict[str, Any]]:
    metrics = {key: dict(value) for key, value in FALLBACK_RUNS.items()}
    for key, summary_path in SUMMARY_PATHS.items():
        summary = _read_json(summary_path)
        if summary is None:
            results_path = RESULTS_PATHS.get(key)
            if results_path is not None:
                summary = _summarize_results(results_path)
        if summary is None:
            continue
        metrics[key].update(
            {
                "total_examples": summary.get("total_examples", metrics[key]["total_examples"]),
                "valid": summary.get("valid", metrics[key]["valid"]),
                "correct": summary.get("correct", metrics[key]["correct"]),
                "valid_rate": summary.get("valid_rate", metrics[key]["valid_rate"]),
                "accuracy": summary.get("accuracy", metrics[key]["accuracy"]),
                "clean_rate": summary.get("clean_rate", metrics[key]["clean_rate"]),
                "score_when_cleaned": summary.get(
                    "score_when_cleaned",
                    summary.get("average_score", metrics[key]["score_when_cleaned"]),
                ),
            }
        )
    return metrics


def _svg_frame(width: int, height: int, title: str, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{title}</title>
  <desc id="desc">Learning to Reset figure generated from local metrics.</desc>
  <rect width="{width}" height="{height}" fill="#f8fafc"/>
  <text x="32" y="40" fill="#0f172a" font-family="Arial, sans-serif" font-size="22" font-weight="700">{title}</text>
{body}
</svg>
"""


def _bar_chart(title: str, rows: Iterable[tuple[str, float, str]], *, maximum: float = 1.0) -> str:
    rows = list(rows)
    width = 860
    height = 110 + len(rows) * 58
    body = []
    for index, (label, value, color) in enumerate(rows):
        y = 82 + index * 58
        bar_width = max(0, min(1, value / maximum)) * 560
        body.append(f'  <text x="32" y="{y + 22}" fill="#334155" font-family="Arial, sans-serif" font-size="15">{label}</text>')
        body.append(f'  <rect x="230" y="{y}" width="560" height="28" fill="#e2e8f0"/>')
        body.append(f'  <rect x="230" y="{y}" width="{bar_width:.1f}" height="28" fill="{color}"/>')
        body.append(f'  <text x="804" y="{y + 20}" fill="#0f172a" font-family="Arial, sans-serif" font-size="14">{value * 100:.1f}%</text>')
    return _svg_frame(width, height, title, "\n".join(body))


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_figures(output_dir: str | Path) -> Dict[str, Any]:
    output_root = Path(output_dir)
    metrics = load_metrics()

    _write(
        output_root / "validity-raw-vs-reset.svg",
        _bar_chart(
            "Validity Recovery: Raw vs Reset-Aware",
            (
                ("raw 0.5B", float(metrics["raw_0_5b"]["valid_rate"]), "#64748b"),
                ("reset-aware 0.5B", float(metrics["reset_0_5b"]["valid_rate"]), "#0f766e"),
                ("grounded fresh-32", float(metrics["grounded_fresh32"]["valid_rate"]), "#2563eb"),
                ("grounded seed=131", float(metrics["grounded_seed131"]["valid_rate"]), "#7c3aed"),
            ),
        ),
    )
    _write(
        output_root / "hard-correctness-gap.svg",
        _bar_chart(
            "Hard Countdown Correctness Gap",
            (
                ("raw 0.5B", float(metrics["raw_0_5b"]["accuracy"]), "#64748b"),
                ("reset-aware 0.5B", float(metrics["reset_0_5b"]["accuracy"]), "#0f766e"),
                ("grounded local runs", float(metrics["grounded_fresh32"]["accuracy"]), "#2563eb"),
                ("paper 1B target", float(metrics["paper_1b"]["accuracy"]), "#b45309"),
            ),
            maximum=0.40,
        ),
    )
    _write(
        output_root / "clean-rate-score.svg",
        _bar_chart(
            "Clean Rate And Score When Cleaned",
            (
                ("reset clean rate", float(metrics["reset_0_5b"]["clean_rate"]), "#0f766e"),
                ("reset score when cleaned", float(metrics["reset_0_5b"]["score_when_cleaned"]), "#14b8a6"),
                ("grounded clean rate", float(metrics["grounded_fresh32"]["clean_rate"]), "#2563eb"),
                ("grounded score when cleaned", float(metrics["grounded_fresh32"]["score_when_cleaned"]), "#60a5fa"),
            ),
        ),
    )
    _write(
        output_root / "token-budget-sweep.svg",
        _bar_chart(
            "Token Budget Sweep: Correctness Stayed Flat",
            (
                ("128 tokens", 0.0, "#94a3b8"),
                ("384 tokens", 0.0, "#64748b"),
                ("multi-clean x3", 0.0, "#475569"),
            ),
            maximum=0.10,
        ),
    )
    pipeline_body = """
  <g font-family="Arial, sans-serif" font-size="14">
    <rect x="42" y="90" width="150" height="54" fill="#dbeafe" stroke="#1d4ed8"/>
    <text x="117" y="122" text-anchor="middle" fill="#172554">source data</text>
    <rect x="232" y="90" width="150" height="54" fill="#ccfbf1" stroke="#0f766e"/>
    <text x="307" y="122" text-anchor="middle" fill="#134e4a">SFT traces</text>
    <rect x="422" y="90" width="150" height="54" fill="#fef3c7" stroke="#b45309"/>
    <text x="497" y="122" text-anchor="middle" fill="#78350f">eval + mine</text>
    <rect x="612" y="90" width="150" height="54" fill="#ede9fe" stroke="#7c3aed"/>
    <text x="687" y="122" text-anchor="middle" fill="#3b0764">re-SFT/RLOO</text>
    <rect x="802" y="90" width="150" height="54" fill="#fee2e2" stroke="#b91c1c"/>
    <text x="877" y="122" text-anchor="middle" fill="#7f1d1d">report</text>
    <path d="M192 117 H232 M382 117 H422 M572 117 H612 M762 117 H802" stroke="#475569" stroke-width="2"/>
    <text x="42" y="200" fill="#334155">Local path: proves validity recovery at 0.5B. Paper path: repeats the same pipeline at 1B+ on GPU.</text>
  </g>
"""
    _write(output_root / "pipeline.svg", _svg_frame(994, 240, "Reset-Aware Replication Pipeline", pipeline_body))

    summary = {"metrics": metrics, "figures": sorted(path.name for path in output_root.glob("*.svg"))}
    _write(output_root / "figure-summary.json", json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build docs/figures SVG assets.")
    parser.add_argument("--output-dir", default="docs/figures")
    args = parser.parse_args(argv)
    summary = build_figures(args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

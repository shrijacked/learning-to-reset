"""Bootstrap reset-aware trace corpora from positive reference traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from learning_to_reset.data import load_trace_records
from learning_to_reset.trace_curation import normalize_trace


def build_bootstrap_negative_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert a positive trace into a think-only negative trace for clean supervision."""

    normalized = normalize_trace(str(record["raw_trace"]))
    metadata = dict(record.get("metadata", {}))
    metadata.update(
        {
            "bootstrap_source_id": record.get("source_id"),
            "bootstrap_kind": "think_only_negative",
        }
    )
    return {
        "source_id": f"{record.get('source_id', 'trace')}:" "bootstrap-negative",
        "problem": str(record["problem"]),
        "raw_trace": f"<think>\n{normalized.think_text}\n</think>",
        "is_correct": False,
        "metadata": metadata,
    }


def build_bootstrap_trace_corpus(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Expand a positive-only corpus into paired positive and negative trace records."""

    corpus = []
    for record in records:
        positive = {
            "source_id": record.get("source_id"),
            "problem": record.get("problem"),
            "raw_trace": record.get("raw_trace"),
            "is_correct": bool(record.get("is_correct", True)),
            "metadata": dict(record.get("metadata", {})),
        }
        corpus.append(positive)
        corpus.append(build_bootstrap_negative_record(positive))
    return corpus


def write_bootstrap_trace_corpus(
    input_path: str | Path,
    output_path: str | Path,
) -> Dict[str, Any]:
    """Write a paired positive/negative bootstrap corpus to JSONL."""

    records = [
        {
            "source_id": record.source_id,
            "problem": record.problem,
            "raw_trace": record.raw_trace,
            "is_correct": record.is_correct,
            "metadata": dict(record.metadata),
        }
        for record in load_trace_records(input_path)
    ]
    corpus = build_bootstrap_trace_corpus(records)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(record, ensure_ascii=True) for record in corpus)
    if payload:
        payload += "\n"
    destination.write_text(payload, encoding="utf-8")
    return {
        "input_records": len(records),
        "records_written": len(corpus),
        "output_path": str(destination),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap paired positive/negative trace records.")
    parser.add_argument("--input-path", required=True, help="Positive trace JSONL path.")
    parser.add_argument("--output-path", required=True, help="Output JSONL path.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = write_bootstrap_trace_corpus(args.input_path, args.output_path)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

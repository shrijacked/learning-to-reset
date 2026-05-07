"""Build deterministic mixed SFT trace sources."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from learning_to_reset.paper_sources import write_jsonl_records


def _read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{source}: expected each JSONL row to be an object.")
            records.append(payload)
    return records


def _annotate_trace_domain(record: Mapping[str, Any], trace_domain: str) -> dict[str, Any]:
    payload = dict(record)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)
    metadata["trace_domain"] = trace_domain
    payload["metadata"] = metadata
    return payload


def _reference_count_for_ratio(
    *,
    synthetic_count: int,
    available_reference_count: int,
    reference_ratio: float,
) -> int:
    if not 0 <= reference_ratio <= 1:
        raise ValueError("reference_ratio must be between 0 and 1.")
    if reference_ratio == 0 or available_reference_count == 0:
        return 0
    if reference_ratio == 1:
        return available_reference_count
    target = round(synthetic_count * reference_ratio / (1 - reference_ratio))
    return min(available_reference_count, max(1, target))


def build_mixed_trace_source(
    *,
    synthetic_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    reference_ratio: float,
    seed: int = 0,
) -> dict[str, Any]:
    """Write synthetic Countdown traces plus a deterministic reference sample."""

    synthetic_records = _read_jsonl_records(synthetic_path)
    reference_records = _read_jsonl_records(reference_path)
    reference_count = _reference_count_for_ratio(
        synthetic_count=len(synthetic_records),
        available_reference_count=len(reference_records),
        reference_ratio=reference_ratio,
    )

    rng = random.Random(seed)
    if reference_count:
        sampled_reference = [
            reference_records[index]
            for index in sorted(rng.sample(range(len(reference_records)), reference_count))
        ]
    else:
        sampled_reference = []

    mixed_records = [
        _annotate_trace_domain(record, "countdown-synthetic")
        for record in synthetic_records
    ]
    mixed_records.extend(
        _annotate_trace_domain(record, "reference-behavior")
        for record in sampled_reference
    )
    rng.shuffle(mixed_records)
    write_jsonl_records(mixed_records, output_path)

    summary = {
        "synthetic_path": str(Path(synthetic_path)),
        "reference_path": str(Path(reference_path)),
        "output_path": str(Path(output_path)),
        "reference_ratio": reference_ratio,
        "seed": seed,
        "synthetic_rows": len(synthetic_records),
        "reference_rows_available": len(reference_records),
        "reference_rows_sampled": len(sampled_reference),
        "records_written": len(mixed_records),
        "trace_domain": {
            "countdown-synthetic": len(synthetic_records),
            "reference-behavior": len(sampled_reference),
        },
    }
    Path(output_path).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a mixed SFT trace JSONL source.")
    parser.add_argument("--synthetic", required=True, help="Synthetic Countdown trace JSONL.")
    parser.add_argument("--reference", required=True, help="Reference behavior trace JSONL.")
    parser.add_argument("--output-path", required=True, help="Mixed trace JSONL output path.")
    parser.add_argument(
        "--reference-ratio",
        type=float,
        default=0.1,
        help="Target fraction of sampled reference traces in the mixed source.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic sampling and shuffle seed.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_mixed_trace_source(
        synthetic_path=args.synthetic,
        reference_path=args.reference,
        output_path=args.output_path,
        reference_ratio=args.reference_ratio,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

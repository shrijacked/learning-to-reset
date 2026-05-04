"""Merge prepared SFT JSONL (prompt/response rows) with mined trace JSONL.

Mined files from ``failure_recovery_traces`` are TraceRecord-shaped (problem,
raw_trace, …), not ``PromptExample`` JSONL. Concatenating them with
``cat`` breaks ``load_prepared_examples``. This module converts mined traces
through ``prepare_sft_examples`` (including recovery rows) and appends them to
the base SFT corpus.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from learning_to_reset.data import load_trace_records
from learning_to_reset.dataset_prep import write_prompt_examples_jsonl
from learning_to_reset.pipeline import prepare_sft_examples
from learning_to_reset.sft_runtime import load_prepared_examples


def merge_sft_train_with_mined_traces(
    *,
    sft_train_path: str | Path,
    mined_traces_path: str | Path,
    output_path: str | Path,
) -> tuple[int, int, int]:
    """Write combined PromptExample JSONL; returns (base_count, mined_count, total)."""

    base = load_prepared_examples(sft_train_path)
    mined_records = load_trace_records(mined_traces_path)
    mined_examples = prepare_sft_examples(
        mined_records,
        include_recovery_examples=True,
        recovery_repeat=1,
        require_recovery_target_correct=False,
    )
    combined = tuple(base) + mined_examples
    write_prompt_examples_jsonl(combined, Path(output_path))
    return len(base), len(mined_examples), len(combined)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--sft-train",
        required=True,
        help="Prepared SFT train JSONL (prompt/response/metadata per line).",
    )
    parser.add_argument(
        "--mined-traces",
        required=True,
        help="Mined recovery traces JSONL from failure_recovery_traces.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for combined PromptExample JSONL.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    base_n, mined_n, total = merge_sft_train_with_mined_traces(
        sft_train_path=args.sft_train,
        mined_traces_path=args.mined_traces,
        output_path=args.output,
    )
    print(
        f"[merge_sft_corpus] wrote {total} examples ({base_n} from --sft-train, "
        f"{mined_n} from mined traces) -> {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

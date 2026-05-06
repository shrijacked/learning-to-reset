"""Validate prepared SFT corpora and report PromptExample composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from learning_to_reset.sft_schema import build_sft_corpus_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate PromptExample JSONL files and report SFT corpus composition."
    )
    parser.add_argument(
        "--path",
        action="append",
        required=True,
        help="Prepared SFT JSONL path to validate. May be supplied multiple times.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON report path. Defaults to printing the report.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    reports = [build_sft_corpus_report(path) for path in args.path]
    payload = {"files": reports}
    rendered = json.dumps(payload, indent=2, ensure_ascii=True)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

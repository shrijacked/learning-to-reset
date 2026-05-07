"""Strict PromptExample JSONL schema validation and composition reporting."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping


PROMPT_EXAMPLE_KEYS = frozenset({"prompt", "response", "metadata"})


class PromptExampleSchemaError(ValueError):
    """Raised when a prepared SFT row is not PromptExample-shaped."""


def _row_label(path: str | Path, row_number: int) -> str:
    return f"{Path(path)} row {row_number}"


def validate_prompt_example_payload(
    payload: Any,
    *,
    path: str | Path,
    row_number: int,
) -> Mapping[str, Any]:
    """Validate one decoded PromptExample JSON row and return it."""

    label = _row_label(path, row_number)
    if not isinstance(payload, Mapping):
        raise PromptExampleSchemaError(f"{label}: expected JSON object.")

    extra_keys = sorted(set(payload) - PROMPT_EXAMPLE_KEYS)
    if extra_keys:
        raise PromptExampleSchemaError(
            f"{label}: unexpected top-level keys {extra_keys!r}; "
            f"allowed keys are {sorted(PROMPT_EXAMPLE_KEYS)!r}."
        )

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise PromptExampleSchemaError(
            f"{label}: required key 'prompt' must be a non-empty string."
        )

    if "response" in payload and not isinstance(payload["response"], str):
        raise PromptExampleSchemaError(
            f"{label}: optional key 'response' must be a string when present."
        )

    if "metadata" in payload and not isinstance(payload["metadata"], Mapping):
        raise PromptExampleSchemaError(
            f"{label}: optional key 'metadata' must be an object when present."
        )

    return payload


def iter_prompt_example_payloads(
    path: str | Path,
    *,
    strict: bool = True,
):
    """Yield decoded non-blank JSONL rows, optionally validating each row."""

    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PromptExampleSchemaError(
                    f"{source} row {row_number}: invalid JSON ({exc.msg})."
                ) from exc
            if strict:
                payload = validate_prompt_example_payload(
                    payload,
                    path=source,
                    row_number=row_number,
                )
            yield row_number, payload


def build_sft_corpus_report(path: str | Path) -> dict[str, Any]:
    """Return schema-valid composition stats for a prepared SFT JSONL file."""

    total_rows = 0
    rows_with_clean = 0
    rows_with_answer = 0
    trace_domains: Counter[str] = Counter()

    for _, payload in iter_prompt_example_payloads(path, strict=True):
        total_rows += 1
        prompt = str(payload["prompt"])
        response = str(payload.get("response", ""))
        metadata = payload.get("metadata", {})
        if "<clean>" in prompt or "<clean>" in response:
            rows_with_clean += 1
        if "<answer>" in prompt or "<answer>" in response:
            rows_with_answer += 1
        trace_domain = metadata.get("trace_domain", "unknown")
        trace_domains[str(trace_domain)] += 1

    return {
        "path": str(Path(path)),
        "total_rows": total_rows,
        "trace_domain": dict(sorted(trace_domains.items())),
        "rows_with_clean": rows_with_clean,
        "rows_with_answer": rows_with_answer,
    }

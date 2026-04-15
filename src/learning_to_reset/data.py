"""Dataset models and loaders for the reset-aware baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


PathLike = Union[str, Path]


@dataclass(frozen=True)
class TraceRecord:
    """A single expert-trace record used for SFT preprocessing."""

    problem: str
    raw_trace: str
    is_correct: bool
    source_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CountdownSample:
    """A single Countdown-style arithmetic sample."""

    numbers: Tuple[int, ...]
    target: int
    question: str
    source_id: Optional[str] = None
    solution: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _read_records(path: PathLike) -> List[Dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        records = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    if source.suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("records", "items", "data", "samples"):
                if key in payload and isinstance(payload[key], list):
                    return payload[key]
            return [payload]

    raise ValueError(f"Unsupported data file format: {source}")


def _pick_value(record: Dict[str, Any], keys: Sequence[str], *, required: bool = True) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    if required:
        raise ValueError(f"Missing required keys {keys} in record: {record}")
    return None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"Could not coerce value to bool: {value!r}")


def _coerce_int_list(value: Any) -> Tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    if isinstance(value, str):
        parts = [part for part in re.split(r"[\s,]+", value.strip()) if part]
        return tuple(int(part) for part in parts)
    raise ValueError(f"Could not coerce value to integer list: {value!r}")


def _collect_metadata(raw: Mapping[str, Any], *, excluded_keys: Sequence[str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    nested_metadata = raw.get("metadata")
    if isinstance(nested_metadata, Mapping):
        metadata.update(dict(nested_metadata))

    for key, value in raw.items():
        if key in excluded_keys or key == "metadata":
            continue
        metadata[key] = value
    return metadata


def render_countdown_question(numbers: Sequence[int], target: int) -> str:
    number_text = ", ".join(str(number) for number in numbers)
    return (
        f"Use the numbers {number_text} to reach {target}. "
        "Use basic arithmetic operations and each number at most once."
    )


def load_trace_records(path: PathLike) -> Tuple[TraceRecord, ...]:
    """Load trace records from JSON or JSONL with flexible field aliases."""

    records = []
    for raw in _read_records(path):
        problem = str(_pick_value(raw, ("problem", "question", "prompt")))
        raw_trace = str(_pick_value(raw, ("raw_trace", "trace", "response")))
        is_correct = _coerce_bool(_pick_value(raw, ("is_correct", "correct", "label")))
        source_id = _pick_value(raw, ("source_id", "id", "name"), required=False)
        metadata = _collect_metadata(
            raw,
            excluded_keys=(
                "problem",
                "question",
                "prompt",
                "raw_trace",
                "trace",
                "response",
                "is_correct",
                "correct",
                "label",
                "source_id",
                "id",
                "name",
            ),
        )
        records.append(
            TraceRecord(
                problem=problem,
                raw_trace=raw_trace,
                is_correct=is_correct,
                source_id=str(source_id) if source_id is not None else None,
                metadata=metadata,
            )
        )
    return tuple(records)


def load_countdown_samples(path: PathLike) -> Tuple[CountdownSample, ...]:
    """Load Countdown-style arithmetic samples from JSON or JSONL."""

    samples = []
    for raw in _read_records(path):
        numbers = _coerce_int_list(_pick_value(raw, ("numbers", "digits", "values")))
        target = int(_pick_value(raw, ("target", "goal", "answer_target")))
        question = _pick_value(raw, ("question", "prompt"), required=False)
        solution = _pick_value(raw, ("solution", "expression", "answer"), required=False)
        source_id = _pick_value(raw, ("source_id", "id", "name"), required=False)
        metadata = _collect_metadata(
            raw,
            excluded_keys=(
                "numbers",
                "digits",
                "values",
                "target",
                "goal",
                "answer_target",
                "question",
                "prompt",
                "solution",
                "expression",
                "answer",
                "source_id",
                "id",
                "name",
            ),
        )
        samples.append(
            CountdownSample(
                numbers=numbers,
                target=target,
                question=str(question) if question is not None else render_countdown_question(numbers, target),
                source_id=str(source_id) if source_id is not None else None,
                solution=str(solution) if solution is not None else None,
                metadata=metadata,
            )
        )
    return tuple(samples)

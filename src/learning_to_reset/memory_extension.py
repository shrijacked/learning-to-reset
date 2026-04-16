"""Recall-aware memory utilities for the context-management extension track."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence, Tuple

from learning_to_reset.context_manager import build_retry_prompt


@dataclass(frozen=True)
class MemoryEntry:
    """One explicit memory item that can be recalled by later prompts."""

    content: str
    tags: Tuple[str, ...] = ()
    source_id: str | None = None


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def _parse_tags(raw_attributes: str) -> Tuple[str, ...]:
    match = re.search(r"""tags\s*=\s*["']([^"']*)["']""", raw_attributes)
    if match is None:
        return ()

    tags = []
    for tag in match.group(1).split(","):
        normalized = _normalize_text(tag)
        if normalized:
            tags.append(normalized)
    return tuple(tags)


def extract_memory_writes(
    response: str,
    *,
    source_id: str | None = None,
) -> Tuple[MemoryEntry, ...]:
    """Extract explicit memory writes from `<memory>...</memory>` blocks."""

    pattern = re.compile(r"<memory(?P<attrs>[^>]*)>(?P<body>.*?)</memory>", re.DOTALL)
    entries = []
    for match in pattern.finditer(response):
        content = _normalize_text(match.group("body"))
        if not content:
            continue
        entries.append(
            MemoryEntry(
                content=content,
                tags=_parse_tags(match.group("attrs")),
                source_id=source_id,
            )
        )
    return tuple(entries)


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def _entry_tokens(entry: MemoryEntry) -> set[str]:
    tokens = set(_tokens(entry.content))
    for tag in entry.tags:
        tokens.update(_tokens(tag))
    return tokens


class RecallMemoryStore:
    """Small deterministic memory store for recall-aware extension experiments."""

    def __init__(self, entries: Iterable[MemoryEntry] = ()) -> None:
        self._entries = list(entries)

    @property
    def entries(self) -> Tuple[MemoryEntry, ...]:
        return tuple(self._entries)

    def write(
        self,
        content: str,
        *,
        tags: Sequence[str] = (),
        source_id: str | None = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            content=_normalize_text(content),
            tags=tuple(_normalize_text(tag) for tag in tags if _normalize_text(tag)),
            source_id=source_id,
        )
        if entry.content:
            self._entries.append(entry)
        return entry

    def extend_from_response(
        self,
        response: str,
        *,
        source_id: str | None = None,
    ) -> Tuple[MemoryEntry, ...]:
        entries = extract_memory_writes(response, source_id=source_id)
        self._entries.extend(entries)
        return entries

    def recall(self, query: str, *, limit: int = 3) -> Tuple[MemoryEntry, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive.")

        query_tokens = _tokens(query)
        scored_entries = []
        for index, entry in enumerate(self._entries):
            score = len(query_tokens & _entry_tokens(entry))
            if score > 0:
                scored_entries.append((score, index, entry))

        scored_entries.sort(key=lambda item: (-item[0], item[1]))
        return tuple(entry for _, _, entry in scored_entries[:limit])


def build_recall_prompt(
    *,
    question: str,
    base_instructions: str,
    recalled_entries: Sequence[MemoryEntry],
) -> str:
    """Build a retry prompt augmented with explicit recalled memory entries."""

    prompt = build_retry_prompt(question, base_instructions)
    entries = tuple(entry for entry in recalled_entries if entry.content.strip())
    if not entries:
        return prompt

    memory_block = "\n".join(f"- {entry.content}" for entry in entries)
    return (
        f"{prompt}\n\n"
        "Relevant recalled memory:\n"
        f"{memory_block}"
    )

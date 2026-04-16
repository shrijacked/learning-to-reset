"""Recall-aware memory utilities for the context-management extension track."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Optional, Sequence, Tuple

from learning_to_reset.context_manager import (
    build_initial_prompt,
    build_retry_prompt,
    extract_answer_text,
    response_requests_clean_retry,
)
from learning_to_reset.data import CountdownSample
from learning_to_reset.rollout_runtime import RewardBreakdown, compute_countdown_reward


@dataclass(frozen=True)
class MemoryEntry:
    """One explicit memory item that can be recalled by later prompts."""

    content: str
    tags: Tuple[str, ...] = ()
    source_id: str | None = None


@dataclass(frozen=True)
class MemoryCleanSegment:
    """One generated segment inside a memory-aware clean interaction."""

    prompt: str
    response: str
    requested_clean: bool
    recalled_entries: Tuple[MemoryEntry, ...]
    memory_writes: Tuple[MemoryEntry, ...]


@dataclass(frozen=True)
class ManagedMemoryCleanGeneration:
    """A bounded clean interaction with explicit memory writes and recall."""

    segments: Tuple[MemoryCleanSegment, ...]
    clean_count: int
    max_cleans: int
    budget_exhausted: bool
    final_response: str
    final_answer: Optional[str]
    memory_entries: Tuple[MemoryEntry, ...]


@dataclass(frozen=True)
class MemoryCleanTrajectory:
    """Reward-attached version of a memory-aware clean interaction."""

    managed: ManagedMemoryCleanGeneration
    segment_rewards: Tuple[RewardBreakdown, ...]
    clean_penalty: float
    adjusted_total_reward: float


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


def manage_memory_clean_cycles(
    *,
    question: str,
    base_instructions: str,
    clean_instructions: str,
    responses: Sequence[str],
    max_cleans: int,
    recall_limit: int = 3,
    clean_token: str = "<clean>",
    initial_memory: Sequence[MemoryEntry] = (),
) -> ManagedMemoryCleanGeneration:
    """Allow bounded cleaning while writing and recalling explicit memory entries."""

    if max_cleans < 0:
        raise ValueError("max_cleans must be non-negative.")
    if recall_limit <= 0:
        raise ValueError("recall_limit must be positive.")
    if not responses:
        raise ValueError("At least one response is required.")

    store = RecallMemoryStore(initial_memory)
    initial_prompt = build_initial_prompt(question, base_instructions, clean_instructions)
    segments = []
    clean_count = 0

    for index, response in enumerate(responses):
        recalled_entries = () if index == 0 else store.recall(question, limit=recall_limit)
        prompt = (
            initial_prompt
            if index == 0
            else build_recall_prompt(
                question=question,
                base_instructions=base_instructions,
                recalled_entries=recalled_entries,
            )
        )
        requested_clean = response_requests_clean_retry(response, clean_token=clean_token)
        memory_writes = store.extend_from_response(
            response,
            source_id=f"segment-{index}",
        )
        segments.append(
            MemoryCleanSegment(
                prompt=prompt,
                response=response,
                requested_clean=requested_clean,
                recalled_entries=recalled_entries,
                memory_writes=memory_writes,
            )
        )

        if not requested_clean:
            final_answer = extract_answer_text(response)
            return ManagedMemoryCleanGeneration(
                segments=tuple(segments),
                clean_count=clean_count,
                max_cleans=max_cleans,
                budget_exhausted=False,
                final_response=response,
                final_answer=final_answer,
                memory_entries=store.entries,
            )

        if clean_count >= max_cleans:
            return ManagedMemoryCleanGeneration(
                segments=tuple(segments),
                clean_count=clean_count,
                max_cleans=max_cleans,
                budget_exhausted=True,
                final_response=response,
                final_answer=extract_answer_text(response),
                memory_entries=store.entries,
            )

        clean_count += 1
        if index == len(responses) - 1:
            raise ValueError("A follow-up response is required while clean budget remains.")

    raise RuntimeError("Memory-aware clean management finished without a terminal response.")


def build_memory_clean_trajectory(
    *,
    question: str,
    sample: CountdownSample,
    responses: Sequence[str],
    max_cleans: int,
    clean_step_penalty: float = 0.05,
    base_instructions: str,
    clean_instructions: str,
    recall_limit: int = 3,
    initial_memory: Sequence[MemoryEntry] = (),
) -> MemoryCleanTrajectory:
    """Compute reward accounting for the memory-aware clean extension path."""

    managed = manage_memory_clean_cycles(
        question=question,
        base_instructions=base_instructions,
        clean_instructions=clean_instructions,
        responses=responses,
        max_cleans=max_cleans,
        recall_limit=recall_limit,
        initial_memory=initial_memory,
    )
    segment_rewards = tuple(
        compute_countdown_reward(segment.response, sample)
        for segment in managed.segments
    )
    clean_penalty = clean_step_penalty * managed.clean_count
    adjusted_total_reward = segment_rewards[-1].total_reward - clean_penalty
    return MemoryCleanTrajectory(
        managed=managed,
        segment_rewards=segment_rewards,
        clean_penalty=clean_penalty,
        adjusted_total_reward=adjusted_total_reward,
    )

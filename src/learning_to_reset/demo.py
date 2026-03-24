"""Runnable baseline demo for the current context-reset mechanics."""

from __future__ import annotations

from dataclasses import dataclass

from learning_to_reset.context_manager import manage_single_clean_cycle
from learning_to_reset.rloo import TrajectorySample, compute_modified_rloo_terms
from learning_to_reset.trace_curation import curate_trace, normalize_trace


@dataclass(frozen=True)
class DemoSnapshot:
    """Structured values used by the text demo report."""

    normalized_think: str
    normalized_answer: str
    curated_uses_clean: bool
    curated_response: str
    final_answer: str
    rloo_normalization: int
    second_advantage: float


def build_demo_snapshot() -> DemoSnapshot:
    """Build a deterministic walkthrough of the implemented baseline mechanics."""

    raw_trace = """
    ignored prefix
    <think>
    First attempt.
    </think>
    <answer>21</answer>
    <think>
    Second attempt.
    </think>
    <answer>42</answer>
    ignored suffix
    """
    normalized = normalize_trace(raw_trace)

    curated = curate_trace(
        raw_trace="""
        <think>
        This path is failing.
        </think>
        <answer>17</answer>
        """,
        is_correct=False,
        base_instructions="Solve the problem carefully.",
        clean_instructions="If your search becomes confusing, emit <clean>.",
    )

    managed = manage_single_clean_cycle(
        question="Make 55 from 61, 63, 57",
        base_instructions="Solve carefully.",
        clean_instructions="Emit <clean> if needed.",
        initial_response="<think>Confusing search.</think><clean>",
        retry_response="<think>Fresh start.</think><answer>55</answer>",
    )

    rloo = compute_modified_rloo_terms(
        [
            TrajectorySample(initial_reward=0.2, initial_length=2),
            TrajectorySample(
                initial_reward=0.1,
                initial_length=4,
                retry_reward=1.0,
                retry_length=5,
            ),
            TrajectorySample(
                initial_reward=0.4,
                initial_length=3,
                retry_reward=0.0,
                retry_length=6,
            ),
        ]
    )

    return DemoSnapshot(
        normalized_think=normalized.think_text,
        normalized_answer=normalized.answer_text or "",
        curated_uses_clean=curated.uses_clean,
        curated_response=curated.response,
        final_answer=managed.final_answer or "",
        rloo_normalization=rloo.normalization,
        second_advantage=rloo.terms[1].advantage,
    )


def build_demo_report() -> str:
    """Render the baseline walkthrough into a readable terminal report."""

    snapshot = build_demo_snapshot()
    sections = [
        "Learning to Reset Baseline Demo",
        "",
        "Trace Preparation",
        f"- Normalized think block: {snapshot.normalized_think}",
        f"- Final normalized answer: {snapshot.normalized_answer}",
        f"- Curated incorrect trace uses reset: {snapshot.curated_uses_clean}",
        f"- Curated response tail: {snapshot.curated_response.splitlines()[-1]}",
        "",
        "Reset Flow",
        "- Initial response emits <clean>.",
        f"- Final answer: {snapshot.final_answer}",
        "",
        "Reward Terms",
        f"- Batch normalization term: {snapshot.rloo_normalization}",
        f"- Second trajectory advantage: {snapshot.second_advantage:.2f}",
    ]
    return "\n".join(sections)


def main() -> None:
    """Print the baseline report to stdout."""

    print(build_demo_report())


if __name__ == "__main__":
    main()

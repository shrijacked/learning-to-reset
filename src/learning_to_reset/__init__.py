"""Core utilities for Learning to Reset."""

from learning_to_reset.context_manager import (
    ManagedGeneration,
    build_initial_prompt,
    build_retry_prompt,
    extract_answer_text,
    manage_single_clean_cycle,
)
from learning_to_reset.rloo import (
    ModifiedRLOOResult,
    TrajectorySample,
    TrajectoryTerm,
    compute_leave_one_out_advantages,
    compute_modified_rloo_terms,
    compute_total_reward,
)
from learning_to_reset.trace_curation import (
    CuratedTrace,
    NormalizedTrace,
    curate_trace,
    normalize_trace,
)

__all__ = [
    "CuratedTrace",
    "ManagedGeneration",
    "ModifiedRLOOResult",
    "NormalizedTrace",
    "TrajectorySample",
    "TrajectoryTerm",
    "build_initial_prompt",
    "build_retry_prompt",
    "compute_leave_one_out_advantages",
    "compute_modified_rloo_terms",
    "compute_total_reward",
    "curate_trace",
    "extract_answer_text",
    "manage_single_clean_cycle",
    "normalize_trace",
]

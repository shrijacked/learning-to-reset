"""Core utilities for Learning to Reset."""

from learning_to_reset.context_manager import (
    ManagedGeneration,
    build_initial_prompt,
    build_retry_prompt,
    extract_answer_text,
    manage_single_clean_cycle,
)
from learning_to_reset.data import CountdownSample, TraceRecord, load_countdown_samples, load_trace_records
from learning_to_reset.dataset_prep import (
    export_prepared_datasets,
    prepare_countdown_split,
    prepare_trace_split,
    serialize_prompt_example,
    write_prompt_examples_jsonl,
)
from learning_to_reset.demo import DemoSnapshot, build_demo_report, build_demo_snapshot
from learning_to_reset.pipeline import (
    DatasetSplit,
    PromptBatch,
    batch_prompt_examples,
    prepare_countdown_examples,
    prepare_sft_examples,
    split_sequence,
)
from learning_to_reset.prompts import (
    PromptExample,
    build_countdown_prompt,
    build_reasoning_prompt,
    build_sft_training_example,
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
    "CountdownSample",
    "DatasetSplit",
    "DemoSnapshot",
    "export_prepared_datasets",
    "ManagedGeneration",
    "ModifiedRLOOResult",
    "NormalizedTrace",
    "PromptBatch",
    "PromptExample",
    "TraceRecord",
    "TrajectorySample",
    "TrajectoryTerm",
    "batch_prompt_examples",
    "build_countdown_prompt",
    "build_initial_prompt",
    "build_retry_prompt",
    "build_demo_report",
    "build_demo_snapshot",
    "build_reasoning_prompt",
    "build_sft_training_example",
    "compute_leave_one_out_advantages",
    "compute_modified_rloo_terms",
    "compute_total_reward",
    "curate_trace",
    "extract_answer_text",
    "load_countdown_samples",
    "load_trace_records",
    "manage_single_clean_cycle",
    "normalize_trace",
    "prepare_countdown_examples",
    "prepare_countdown_split",
    "prepare_sft_examples",
    "prepare_trace_split",
    "serialize_prompt_example",
    "split_sequence",
    "write_prompt_examples_jsonl",
]

# Paper claims traceability

This document maps every mechanism described in `main.pdf` Sections 3 and 4 to
the file(s) and test(s) that implement it in this repository. Use it as the
authoritative reference when verifying that a paper-promised behaviour is
actually shipped.

When a row is marked **partial** the file implements the mechanism but not at
the paper's full data scale; the paper-replicate runbook
(`docs/paper-replication.md`) covers the path to bring it to full scale.

## Section 3 -- Method

| Paper claim | Section / Figure | Implementation | Tests |
| --- | --- | --- | --- |
| Curate expert traces: keep correct ones, rewrite incorrect ones into recovery (think + clean) examples | Section 3.2.1, Figure 3 | `src/learning_to_reset/trace_curation.py`, `src/learning_to_reset/synthetic_countdown_traces.py`, `src/learning_to_reset/failure_recovery_traces.py` | `tests/test_trace_curation.py`, `tests/test_synthetic_countdown_traces.py`, `tests/test_failure_recovery_traces.py` |
| Recovery trace styles (walkthrough, verification, contrastive) | Section 3.2.1 | `synthetic_countdown_traces.build_recovery_response`, `build_verified_recovery_response`, `build_contrastive_recovery_response` | `tests/test_synthetic_countdown_traces.py` |
| Arithmetic-grounded recovery template (extension over the paper to attack substep fabrication) | Section 3.2.1 + status report | `synthetic_countdown_traces.build_grounded_recovery_response` | `tests/test_synthetic_countdown_traces.py` (grounded suite) |
| Deep-verify filter for retry-stage recoveries | Section 3.2.1 (verifier-grounded recoveries) | `pipeline._deep_verify_recovery_text` | `tests/test_pipeline.py::test_require_target_correct_recovery_drops_inconsistent_arithmetic_claims` |
| One-shot context manager: `<clean>` triggers a fresh retry prompt without cleaning instructions | Section 3.2.2, Figure 4 | `src/learning_to_reset/context_manager.py` (`response_requests_clean_retry`, `manage_single_clean_cycle`, `build_initial_prompt`, `build_retry_prompt`) | `tests/test_context_manager.py` |
| Verifier-aware multi-clean retry decoder (extension of the paper's single retry to a budgeted multi-retry decoder) | Section 3.2.2 | `eval_runtime.run_multi_clean_eval_loop`, `eval_runtime.evaluate_with_multi_clean_decoder` | `tests/test_eval_runtime.py::MultiCleanEvalLoopTests` |
| Modified RLOO update with leave-one-out advantage | Section 3.2.3, Eq. 4-6 | `src/learning_to_reset/rloo.py` (`compute_modified_rloo_terms`, `compute_policy_loss`) | `tests/test_rloo.py`, `tests/test_rloo_runtime.py` |
| Normalization by `k + kc` (count of cleaned trajectories in the batch) | Section 3.2.3, Eq. 5 | `rloo.compute_modified_rloo_terms` | `tests/test_rloo.py::test_modified_rloo_terms_match_paper_normalization` |
| Reward = formatting + correctness; policy update uses correctness-only reward | Section 3.2.3, Section 4.1 | `rollout_runtime.compute_countdown_reward`, `rloo_runtime.build_policy_trajectory_sample` | `tests/test_rollout_runtime.py`, `tests/test_rloo_runtime.py` |
| Bounded multi-clean extension | Section 3.2.4 / Discussion | `multi_clean_extension.manage_bounded_clean_cycles`, `build_multi_clean_trajectory` | `tests/test_multi_clean_extension.py` |
| Selective retention extension (`<retain>...</retain>` notes survive a reset) | Section 3.2.4 / Discussion | `multi_clean_extension.manage_selective_retention_clean_cycles`, `build_selective_retention_trajectory` | `tests/test_multi_clean_extension.py` |
| Memory recall extension (`<memory>...</memory>` writes + retrieval) | Section 3.2.4, Figure 1, Conclusion | `src/learning_to_reset/memory_extension.py` | `tests/test_memory_extension.py` |
| Side-by-side comparison of full reset / retention / memory controllers | Discussion / extension evaluation plan | `src/learning_to_reset/extension_comparison.py` | `tests/test_extension_comparison.py` |

## Section 4 -- Experiments

| Paper claim | Section / Figure | Implementation | Tests / Artifact |
| --- | --- | --- | --- |
| Countdown task setup (k=4 inputs, target reach via four ops) | Section 4.1 | `src/learning_to_reset/countdown_solver.py`, `src/learning_to_reset/countdown_verifier.py` | `tests/test_countdown_solver.py`, `tests/test_countdown_verifier.py` |
| Hard-slice export (multiplication/division-heavy held-out evaluation) | Section 4.3.1 | `src/learning_to_reset/countdown_slices.py`, `src/learning_to_reset/paper_dataset_prep.py` | `tests/test_countdown_slices.py`, `tests/test_paper_dataset_prep.py` |
| Hard-focused source generation for training | Section 4.3.1 | `src/learning_to_reset/synthetic_countdown_dataset.py` | `tests/test_synthetic_countdown_dataset.py` |
| Failure mining of hard-eval failures into verified recovery traces (with contamination guard) | Section 4.3.1 | `src/learning_to_reset/failure_recovery_traces.py` (incl. `--exclude-source-ids`) | `tests/test_failure_recovery_traces.py` (`test_build_failure_recovery_trace_records_rejects_eval_overlap`, `test_generate_failure_recovery_trace_corpus_loads_exclusion_files`) |
| Reset-aware SFT runtime against prepared artifacts | Section 4.1 | `src/learning_to_reset/sft_runtime.py` | `tests/test_sft_runtime.py` |
| Reset-aware RLOO runtime | Section 4.1 | `src/learning_to_reset/rloo_runtime.py` | `tests/test_rloo_runtime.py` |
| Clean-aware evaluation runtime (single-shot reset by default) | Section 4 | `src/learning_to_reset/eval_runtime.py` | `tests/test_eval_runtime.py` |
| Clean rate metric (Figure 6 left axis) | Section 4 / Figure 6 | `eval_runtime.evaluate_countdown_outputs`, `rloo_runtime.evaluate_rollout_candidates` | `tests/test_eval_runtime.py::test_evaluate_countdown_outputs_reports_clean_rate_and_score_when_cleaned`, `tests/test_rloo_runtime.py::test_summarize_rollout_candidates_reports_clean_and_score` |
| Score-when-cleaned metric (Figure 6 right axis) | Section 4 / Figure 6 | `evaluate_countdown_outputs`, `evaluate_rollout_candidates` | `tests/test_eval_runtime.py::test_evaluate_countdown_outputs_reports_clean_rate_and_score_when_cleaned`, `tests/test_rloo_runtime.py::test_evaluate_rollout_candidates_reports_score_when_cleaned` |
| Raw vs reset-aware comparison plotted side-by-side | Section 4 | `src/learning_to_reset/compare_eval_results.py` | `tests/test_compare_eval_results.py` |
| Qualitative samples (Figure 7) | Section 4.3.3 / Figure 7 | `src/learning_to_reset/qualitative_export.py` (and `scripts/export_qualitative_samples.py`) | `tests/test_qualitative_export.py` |
| Hard-correctness jump from 13.61% to 36.94% (1B model) | Section 4.3 | **partial** -- code path runs at 1B; replicate via `docs/paper-replication.md`. Local 0.5B pilot ceiling: 1/32 (raw eval) -> 0/32 (grounded SFT) -- see `docs/grounded-recovery-results-2026-04-26.md` | re-runs of `eval_runtime` after replicate path is executed |

## How to use this table

1. When you ship a new mechanism, add a row here (and the corresponding test
   row in `docs/project-plan.md`).
2. When a paper claim is **partial**, link to the runbook that closes the gap
   and document any local ceiling explicitly so the gap is honest.
3. When a paper claim is dropped or replaced (e.g., a new template extending
   the paper's set), keep the original row, mark the disposition, and add the
   replacement row beneath it so the trail is preserved.

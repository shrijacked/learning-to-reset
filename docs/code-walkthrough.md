# Code Walkthrough

## Big Picture

This repository currently implements the baseline mechanics for a reset-aware reasoning workflow. The idea is simple:

1. take expert traces and convert them into a clean training format
2. let the model emit a special `<clean>` token when its reasoning path becomes unproductive
3. treat the final interaction outcome as the reward signal, whether the answer comes from the first try or from a retry after reset

The repository now includes a local baseline runtime over prepared artifacts. In practice that means the repo can prepare data, run SFT, run a reset-aware RLOO loop, and evaluate with the one-shot clean retry path. The first local target-model pilots and verifier-grounded recovery ablations have completed; the strongest current signal is that reset-aware evaluation makes post-clean answers well-formed, while target-correct arithmetic remains the main blocker.

## Repo Map

- `src/learning_to_reset/trace_curation.py`
  - prepares traces for reset-aware supervised fine-tuning
- `src/learning_to_reset/data.py`
  - loads trace and Countdown-style records from JSON/JSONL
- `src/learning_to_reset/prompts.py`
  - builds and parses reasoning prompts with optional clean instructions
- `src/learning_to_reset/pipeline.py`
  - converts records into split-ready prompt examples
- `src/learning_to_reset/dataset_prep.py`
  - exports trainer-ready JSONL artifacts and manifests
- `src/learning_to_reset/context_manager.py`
  - handles the one-shot reset flow
- `src/learning_to_reset/countdown_verifier.py`
  - checks expression legality and whether a response hits the target
- `src/learning_to_reset/synthetic_countdown_traces.py`
  - builds solver-backed Countdown trace records for local fallback training
  - supports walkthrough and verifier-grounded recovery responses after reset
- `src/learning_to_reset/rloo.py`
  - implements the reward math for reset-aware RLOO
- `src/learning_to_reset/rollout_runtime.py`
  - attaches reward breakdowns and trajectory objects to sampled responses
- `src/learning_to_reset/sft_runtime.py`
  - runs supervised fine-tuning on prepared artifacts
- `src/learning_to_reset/eval_runtime.py`
  - runs Countdown evaluation, defaulting to clean-aware retry behavior
  - records the verifier-computed expression value for debugging wrong-target answers
- `src/learning_to_reset/compare_eval_results.py`
  - compares raw one-pass and reset-aware evaluation summaries
  - writes compact JSON and Markdown artifacts for result tables
- `src/learning_to_reset/rloo_runtime.py`
  - runs reset-aware rollouts, computes policy loss, writes metrics, and saves checkpoints
- `src/learning_to_reset/multi_clean_extension.py`
  - implements the bounded multi-clean extension path with a reset budget and per-clean penalty
- `src/learning_to_reset/demo.py`
  - gives a deterministic walkthrough of the current mechanics
- `tests/`
  - verifies each of the baseline pieces independently

## File-by-File Walkthrough

### `trace_curation.py`

This file is the SFT preprocessing layer.

Important pieces:

- `NormalizedTrace`
  - a dataclass holding a single merged `<think>` block and the final `<answer>`
- `CuratedTrace`
  - a dataclass holding the final SFT example after instructions and reset behavior are applied
- `normalize_trace(raw_trace)`
  - extracts all `<think>` and `<answer>` blocks
  - merges multiple think blocks into one reasoning block
  - keeps only the final answer
  - drops everything outside the tagged blocks
- `curate_trace(...)`
  - if the trace is correct, it keeps the normal think-plus-answer format
  - if the trace is incorrect or unproductive, it appends a reset rationale and replaces the final answer with `<clean>`

Why it matters:

- this is where ordinary reasoning traces become reset-aware training examples
- it encodes the idea that some bad traces should teach the model to restart instead of forcing it to keep answering from a confused state

### `context_manager.py`

This file simulates the one-shot reset interaction.

Important pieces:

- `ManagedGeneration`
  - stores the initial prompt, retry prompt, responses, and final answer for one interaction
- `build_initial_prompt(...)`
  - includes both the base instructions and the reset instructions
- `build_retry_prompt(...)`
  - removes the reset instructions for the second attempt
- `extract_answer_text(response)`
  - reads the final `<answer>` block from a response
- `manage_single_clean_cycle(...)`
  - checks whether the first response contains `<clean>`
  - if not, the first response is the final response
  - if yes, the context is treated as reset and a second response becomes the final response

Why it matters:

- this is the runtime logic that gives meaning to `<clean>`
- without this layer, the token is just text and not an actual control action

### `rloo.py`

This file holds the reset-aware reward math.

Important pieces:

- `TrajectorySample`
  - a single sampled interaction containing the first attempt and an optional retry
- `TrajectoryTerm`
  - the per-trajectory reward and scaling terms used for optimization
- `ModifiedRLOOResult`
  - the batch-level result after the full computation
- `compute_total_reward(...)`
  - chooses the retry reward if a retry happened, otherwise the initial reward
- `compute_leave_one_out_advantages(rewards)`
  - computes the leave-one-out baseline and per-sample advantage values
- `compute_modified_rloo_terms(trajectories)`
  - computes interaction rewards
  - computes leave-one-out advantages
  - adjusts the normalization with `k + kc`, where `kc` is the number of cleaned trajectories
  - returns the scaling terms for the initial response and the retry response

Why it matters:

- this is the main RL-specific logic in the repository
- it captures the idea that a successful retry should reinforce both the decision to clean and the answer after the reset

### `data.py`, `prompts.py`, `pipeline.py`, and `dataset_prep.py`

These files turn raw project data into trainer-ready artifacts.

Important pieces:

- `TraceRecord` and `CountdownSample`
  - dataclasses for the two main data sources in the project
- `load_trace_records(...)` and `load_countdown_samples(...)`
  - flexible JSON/JSONL loaders with alias handling
- `build_reasoning_prompt(...)`
  - creates prompts with base instructions, optional clean instructions, and the question
- `parse_reasoning_prompt(...)`
  - recovers the base instructions, clean instructions, and question from a prepared prompt
- `prepare_sft_examples(...)` and `prepare_countdown_examples(...)`
  - convert raw records into prompt examples for SFT and Countdown rollouts
- `export_prepared_datasets(...)`
  - writes split JSONL files plus a manifest for later runtimes

Why they matter:

- these files are the bridge between project data and every training or evaluation stage
- they make the rest of the pipeline deterministic and easier to test

### `countdown_verifier.py`

This file is the reward and evaluation checker for arithmetic outputs.

Important pieces:

- `extract_answer_expression(...)`
  - reads the final `<answer>` expression from a response
- `verify_countdown_expression(...)`
  - checks number usage, expression validity, and target reachability
- `score_countdown_response(...)`
  - packages the verification result for downstream reward logic

Why it matters:

- this is the ground-truth scoring layer used by evaluation and the RL runtime
- it separates reasoning generation from arithmetic correctness checking

### `rollout_runtime.py`, `sft_runtime.py`, `eval_runtime.py`, and `rloo_runtime.py`

These files turn the math and prompt mechanics into runnable training and evaluation entrypoints.

Important pieces:

- `rollout_runtime.py`
  - computes formatting and correctness rewards
  - builds clean-aware trajectory objects for modified RLOO
- `sft_runtime.py`
  - loads prepared SFT JSONL
  - tokenizes prompt/response pairs
  - runs `transformers`-based supervised fine-tuning
- `eval_runtime.py`
  - scores generated Countdown answers
  - defaults to the clean-aware retry path for the paper-style baseline
- `compare_eval_results.py`
  - loads two evaluation `summary.json` files
  - computes candidate-minus-baseline deltas for accuracy, validity, average score, and clean rate
  - writes `comparison.json` and `comparison.md`
- `rloo_runtime.py`
  - samples `k` reset-aware trajectories per prompt
  - computes modified-RLOO scaling
  - assembles policy loss over `y0` and optional `y1`
  - writes per-step metrics and saves checkpoints

Why they matter:

- this is where the repository stops being only a mechanics demo and becomes a runnable baseline
- these entrypoints are what you would use to produce the first actual baseline runs

### `multi_clean_extension.py`

This file contains the first extension module.

Important pieces:

- `manage_bounded_clean_cycles(...)`
  - allows multiple clean requests up to a fixed budget
  - stops once a normal answer appears or the budget is exhausted
- `build_multi_clean_trajectory(...)`
  - computes reward accounting over the bounded interaction
  - subtracts a small penalty for each clean action

Why it matters:

- it keeps the one-shot baseline stable while giving the extension track a tested place to grow
- it lets the project study whether extra resets help or whether the model starts using reset too aggressively

### `demo.py`

This file is a runnable summary of the baseline.

Important pieces:

- `DemoSnapshot`
  - collects the key intermediate values from the current mechanics
- `build_demo_snapshot()`
  - runs one example through trace normalization, trace curation, reset handling, and reward computation
- `build_demo_report()`
  - turns the snapshot into a readable terminal report
- `python -m learning_to_reset`
  - runs the demo through the package entry point in `__main__.py`

Why it matters:

- it gives the team a concrete artifact to show in discussion or Q&A
- it proves the baseline pieces work together conceptually even before full model training exists

## Tests and What They Prove

### `tests/test_trace_curation.py`

Checks that:

- multiple think blocks are merged correctly
- only the final answer is kept
- incorrect traces end with `<clean>`
- correct traces must still have a final answer

### `tests/test_context_manager.py`

Checks that:

- a normal answer path returns the first response directly
- a clean path requires a retry response
- the retry prompt removes the reset instructions
- the final answer is taken from the retry when reset happens

### `tests/test_rloo.py`

Checks that:

- retry reward overrides the first reward when reset happens
- leave-one-out advantages match expected values
- the modified normalization and scaling terms are computed correctly

### `tests/test_demo.py`

Checks that:

- the demo snapshot contains the expected baseline values
- the demo report actually covers trace prep, reset flow, and reward terms

## Jargon Cheat Sheet

- `SFT`
  - supervised fine-tuning; learning directly from example traces
- `trace`
  - the reasoning text produced for a problem
- `context management`
  - controlling what reasoning history stays active for the model
- `<clean>`
  - the special token that triggers a reset of the reasoning context
- `y0`
  - the first generated response
- `y1`
  - the retry response after reset
- `trajectory`
  - one sampled interaction from prompt to final outcome
- `reward`
  - the score assigned to the interaction outcome
- `advantage`
  - how much better or worse one sample is than its baseline
- `RLOO`
  - REINFORCE with Leave-One-Out; a low-variance policy-gradient style update
- `baseline`
  - the currently implemented one-shot reset mechanics
- `extension`
  - future work such as multi-step cleaning, selective retention, and recall-aware memory

## What Is Not in the Code Yet

The repository does not yet include:

- a strong expert-trace source with both productive and unproductive tagged Countdown traces
- a larger target-model run with enough data and compute to produce nontrivial Countdown accuracy
- a scaled raw-versus-reset-aware table on a larger hard Countdown slice
- extension stages beyond bounded multi-step cleaning, especially selective retention and recall-aware memory

So the honest state is:

- the baseline mechanics are implemented
- the local runtime pipeline is in place
- the remaining gap is stronger arithmetic grounding and larger experiment execution

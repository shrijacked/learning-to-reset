# Code Walkthrough

## Big Picture

This repository currently implements the baseline mechanics for a reset-aware reasoning workflow. The idea is simple:

1. take expert traces and convert them into a clean training format
2. let the model emit a special `<clean>` token when its reasoning path becomes unproductive
3. treat the final interaction outcome as the reward signal, whether the answer comes from the first try or from a retry after reset

Right now, the repository is a mechanics-and-testing scaffold, not a full training system. The code is focused on the building blocks that the eventual SFT and RL pipeline will need.

## Repo Map

- `src/learning_to_reset/trace_curation.py`
  - prepares traces for reset-aware supervised fine-tuning
- `src/learning_to_reset/context_manager.py`
  - handles the one-shot reset flow
- `src/learning_to_reset/rloo.py`
  - implements the reward math for reset-aware RLOO
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

- expert-trace ingestion from full datasets
- Countdown dataset adapters
- prompt packing and batch construction for training
- the full SFT loop
- the full reset-aware RLOO loop
- real experiment tracking and metrics

So the honest state is:

- the mechanics are implemented
- the engineering skeleton is solid
- the end-to-end training pipeline is still to be built

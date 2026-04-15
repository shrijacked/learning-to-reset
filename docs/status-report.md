# Status Report

## Current Engineering State

Implemented and verified:

- trace normalization and SFT curation for `<clean>`-aware examples
- one-shot context reset manager for `y0 -> optional <clean> -> y1`
- reset-aware RLOO reward and advantage utilities
- JSON/JSONL dataset loaders for trace and Countdown-style records
- prompt builders for SFT and reset-aware evaluation inputs
- deterministic split and batching helpers for trainer preparation
- JSONL export and CLI preparation path for trainer-ready splits
- Countdown expression verification for generated answers
- SFT runtime, clean-aware evaluation runtime, and clean-aware trajectory assembly
- reset-aware RLOO runtime with local metrics, best-checkpoint saving, and CLI entrypoint
- unit-test coverage for the baseline mechanics
- a tiny-model smoke run through the reset-aware trainer and evaluator
- paper-aligned source adapters for the upstream Countdown datasets
- a paper-aligned artifact preparation path that keeps external train/eval splits separate
- a first real local pilot run on fetched source data
- GitHub repository setup and CI for the test suite

## Working Baseline

The repository now supports the baseline code path end to end over prepared artifacts. It has not yet been run against the project's real datasets and target model checkpoints.

What works today:

- curated expert traces can be transformed into reset-aware training examples
- a reset interaction can be simulated deterministically
- reward propagation for reset-aware training can be inspected and tested
- trace/countdown records can be loaded into structured dataclasses
- reasoning prompts and SFT examples can be assembled from those records
- prepared prompt examples can be split and batched for future trainers
- split SFT/eval artifacts can be exported as JSONL with a manifest
- generated Countdown answers can be checked for legality and target correctness
- a local `transformers` stack can run SFT over prepared artifacts
- a reset-aware RLOO trainer can sample `k` trajectories, apply one-shot retry logic, and optimize with the modified loss
- clean-aware evaluation can retry once after `<clean>` and record score, accuracy, and clean rate
- training runs emit local metrics and checkpoint outputs for inspection
- the repo can fetch and flatten the upstream Countdown train/eval datasets into local JSONL source files
- the repo can prepare paper-aligned artifacts from those real source files
- the target Qwen model can complete a first local SFT checkpoint and a first local reset-aware RL checkpoint on a pilot subset

What is not implemented yet:

- a strong expert-trace source that cleanly provides both productive and unproductive tagged traces for the reset-aware SFT stage
- a larger paper-style run with enough data and compute to produce nontrivial Countdown accuracy
- baseline-versus-reset-aware comparison on a properly sized hard Countdown slice
- extension stages such as multi-step cleaning, selective retention, and recall-aware memory

## Current Pilot Result

The first real local pilot run completed on a tiny fetched subset and a CPU-only setup.

Observed pilot outcome:

- SFT checkpoint trained successfully on 21 train traces and 3 validation traces
- reset-aware RL checkpoint ran successfully on 9 Countdown train prompts and 3 validation prompts
- base, SFT, and reset-aware checkpoints all scored `0/4` on the tiny held-out eval slice

Interpretation:

- the source wiring and runtime pipeline now work on real fetched assets
- the current pilot is still too weak to claim meaningful paper-level performance
- the next blocker is better trace supervision and a stronger run configuration, not missing plumbing

## Remaining Engineering Work

1. Point the preparation command at the real expert-trace and Countdown files.
2. Replace the provisional positive-trace slice with a stronger expert-trace source.
3. Re-run the SFT stage on a larger paper-aligned split.
4. Re-run the reset-aware RLOO stage on top of that checkpoint.
5. Evaluate on hard Countdown examples and compare against the default baseline.
6. Start extension work only after the baseline pipeline produces stable outputs.

## Remaining Non-Engineering Work

- choose the presentation slot in the shared schedule
- convert the technical material into the final five slides
- assign speaking order and rehearse timing for the 8-minute talk plus Q&A

## Recommended Talking Points Right Now

- the technical baseline is no longer just an idea; the core mechanics exist in code and are tested
- the next meaningful milestone is the first paper-aligned run on the real datasets, not another code scaffold
- the most important future direction is stronger context management beyond one-shot reset

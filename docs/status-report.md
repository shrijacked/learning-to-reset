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

What is not implemented yet:

- wiring the preparation command to the real source datasets used by the project
- the first paper-aligned SFT and RLOO runs on the actual target model
- baseline-versus-reset-aware comparison on the real hard Countdown split
- extension stages such as multi-step cleaning, selective retention, and recall-aware memory

## Remaining Engineering Work

1. Point the preparation command at the real expert-trace and Countdown files.
2. Run the first SFT pipeline around the exported splits and the target model.
3. Run the reset-aware RLOO stage on top of that SFT checkpoint.
4. Evaluate on hard Countdown examples and compare against the default baseline.
5. Start extension work only after the baseline pipeline produces stable outputs.

## Remaining Non-Engineering Work

- choose the presentation slot in the shared schedule
- convert the technical material into the final five slides
- assign speaking order and rehearse timing for the 8-minute talk plus Q&A

## Recommended Talking Points Right Now

- the technical baseline is no longer just an idea; the core mechanics exist in code and are tested
- the next meaningful milestone is the first paper-aligned run on the real datasets, not another code scaffold
- the most important future direction is stronger context management beyond one-shot reset

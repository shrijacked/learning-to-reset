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
- unit-test coverage for the baseline mechanics
- GitHub repository setup and CI for the test suite

## Working Baseline

The repository currently supports the baseline mechanics around context reset. It does not yet run full model training or dataset evaluation end to end.

What works today:

- curated expert traces can be transformed into reset-aware training examples
- a reset interaction can be simulated deterministically
- reward propagation for reset-aware training can be inspected and tested
- trace/countdown records can be loaded into structured dataclasses
- reasoning prompts and SFT examples can be assembled from those records
- prepared prompt examples can be split and batched for future trainers
- split SFT/eval artifacts can be exported as JSONL with a manifest
- generated Countdown answers can be checked for legality and target correctness

What is not implemented yet:

- wiring the preparation command to the real source datasets used by the project
- model-generation orchestration around the verifier
- trainer integration on top of the prepared batches
- full SFT training loop
- full RLOO training loop
- experiment tracking for baseline versus reset-aware runs

## Remaining Engineering Work

1. Point the preparation command at the real expert-trace and Countdown files.
2. Build the first SFT pipeline around the exported splits.
3. Add the reset-aware RLOO training loop on top of the SFT checkpoint.
4. Run evaluation on hard Countdown examples and compare against the baseline.
5. Start extension work only after the baseline pipeline produces stable outputs.

## Remaining Non-Engineering Work

- choose the presentation slot in the shared schedule
- convert the technical material into the final five slides
- assign speaking order and rehearse timing for the 8-minute talk plus Q&A

## Recommended Talking Points Right Now

- the technical baseline is no longer just an idea; the core mechanics exist in code and are tested
- the next meaningful milestone is not another slide artifact, but the first end-to-end training and evaluation run
- the most important future direction is stronger context management beyond one-shot reset

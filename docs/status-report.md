# Status Report

## Current Engineering State

Implemented and verified:

- trace normalization and SFT curation for `<clean>`-aware examples
- one-shot context reset manager for `y0 -> optional <clean> -> y1`
- reset-aware RLOO reward and advantage utilities
- unit-test coverage for the baseline mechanics
- GitHub repository setup and CI for the test suite

## Working Baseline

The repository currently supports the baseline mechanics around context reset. It does not yet run full model training or dataset evaluation end to end.

What works today:

- curated expert traces can be transformed into reset-aware training examples
- a reset interaction can be simulated deterministically
- reward propagation for reset-aware training can be inspected and tested

What is not implemented yet:

- expert-trace dataset ingestion from the full source files
- Countdown dataset adapters and evaluation harness
- prompt assembly for training and evaluation runs
- full SFT training loop
- full RLOO training loop
- experiment tracking for baseline versus reset-aware runs

## Remaining Engineering Work

1. Add expert-trace and Countdown dataset loaders.
2. Connect the existing mechanics to prompt assembly and batching code.
3. Build the first SFT pipeline around the curated traces.
4. Add the reset-aware RLOO training loop on top of the SFT checkpoint.
5. Run evaluation on hard Countdown examples and compare against the baseline.
6. Start extension work only after the baseline pipeline produces stable outputs.

## Remaining Non-Engineering Work

- choose the presentation slot in the shared schedule
- convert the technical material into the final five slides
- assign speaking order and rehearse timing for the 8-minute talk plus Q&A

## Recommended Talking Points Right Now

- the technical baseline is no longer just an idea; the core mechanics exist in code and are tested
- the next meaningful milestone is not another slide artifact, but the first end-to-end training and evaluation run
- the most important future direction is stronger context management beyond one-shot reset

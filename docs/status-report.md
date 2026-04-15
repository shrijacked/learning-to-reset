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
- a deterministic Countdown solver and synthetic Countdown-aligned fallback trace generator
- automatic resolution of nested `best-checkpoint` and `final-checkpoint` model outputs
- a first bounded multi-step cleaning extension with clean-budget and clean-penalty support
- retry-stage recovery augmentation for fallback SFT preparation
- a recovery-balance control for repeating retry-stage examples during fallback dataset prep
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
- the repo can synthesize Countdown-aligned fallback traces locally when a stronger trace source is unavailable
- the target Qwen model can complete a first local SFT checkpoint and a first local reset-aware RL checkpoint on a pilot subset
- evaluation can load trainer output roots directly even when the actual model lives inside `best-checkpoint` or `final-checkpoint`
- fallback SFT preparation can append retry-stage recovery examples and rebalance them explicitly

What is not implemented yet:

- a strong paper-native expert-trace source that cleanly provides both productive and unproductive tagged traces for the reset-aware SFT stage
- a larger paper-style run with enough data and compute to produce nontrivial Countdown accuracy
- baseline-versus-reset-aware comparison on a properly sized hard Countdown slice
- extension stages beyond bounded multi-step cleaning, especially selective retention and recall-aware memory

## Current Pilot Result

The first real local pilot run completed on a tiny fetched subset and a CPU-only setup.

Observed pilot outcome:

- SFT checkpoint trained successfully on 21 train traces and 3 validation traces
- reset-aware RL checkpoint ran successfully on 9 Countdown train prompts and 3 validation prompts
- base, SFT, and reset-aware checkpoints all scored `0/4` on the tiny held-out eval slice
- a Countdown-aligned synthetic fallback trace run also completed end to end on the same local slice
- the synthetic SFT checkpoint still scored `0/4`, but it moved `clean_rate` to `1.0`, showing the model learned the reset action more strongly than the recovery action
- the reset-aware RL stage on top of that synthetic SFT checkpoint also remained `0/4`
- a stronger fallback SFT run with repeated retry-recovery supervision reached `valid_rate = 1.0` and `average_score = 0.1` on the held-out slice
- the corresponding reset-aware RL run reached `validation_accuracy = 1/3` and `validation_average_score = 0.4` on its local validation slice, while the held-out slice stayed at `0/4` correct and `4/4` valid

Interpretation:

- the source wiring and runtime pipeline now work on real fetched assets
- the current pilot is still too weak to claim meaningful paper-level performance
- the main remaining baseline blocker is no longer formatting or reset behavior; it is target-correctness after the clean step, plus larger compute/data
- the first extension module is now implemented separately from the baseline path, so future work can extend beyond one-shot cleaning without destabilizing the paper baseline

## Remaining Engineering Work

1. Replace the fallback trace slice with a stronger paper-native expert-trace source.
2. Strengthen the fallback recovery path further so the post-clean retry stage learns target-correct expressions, not just legal ones.
3. Re-run the SFT stage on a larger paper-aligned split.
4. Re-run the reset-aware RLOO stage on top of that checkpoint.
5. Evaluate on hard Countdown examples and compare against the default baseline.
6. Extend the current multi-clean module toward selective retention and recall-aware memory.

## Remaining Non-Engineering Work

- choose the presentation slot in the shared schedule
- convert the technical material into the final five slides
- assign speaking order and rehearse timing for the 8-minute talk plus Q&A

## Recommended Talking Points Right Now

- the technical baseline is no longer just an idea; the core mechanics exist in code and are tested
- the next meaningful milestone is stronger target-correct recovery after the clean step, ideally from a paper-native or stronger Countdown-aligned trace source
- the model now reliably reaches legal post-clean expressions on the held-out slice, so the remaining gap is correctness rather than formatting
- the most important future direction remains stronger context management beyond one-shot reset

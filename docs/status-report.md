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
- paper-aligned source adapters for the upstream reference-trace dataset
- automatic bootstrap from positive reference traces into a paired reset-aware trace corpus
- a paper-aligned artifact preparation path that keeps external train/eval splits separate
- a first real local pilot run on fetched source data
- a live smoke-verified paper-source flow from fetch -> trace bootstrap -> artifact preparation
- a deterministic Countdown solver and synthetic Countdown-aligned fallback trace generator
- multi-solution synthetic Countdown supervision and step-by-step arithmetic walkthrough traces
- automatic resolution of nested `best-checkpoint` and `final-checkpoint` model outputs
- a first bounded multi-step cleaning extension with clean-budget and clean-penalty support
- retry-stage recovery augmentation for fallback SFT preparation
- a recovery-balance control for repeating retry-stage examples during fallback dataset prep
- GitHub repository setup and CI for the test suite

## Working Baseline

The repository now supports the baseline code path end to end over prepared artifacts. The real source-data path is now wired and smoke-verified locally, but the larger target-model experiments are still outstanding.

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
- the repo can fetch positive reference traces and expand them into a paired reset-aware trace corpus
- the repo can prepare paper-aligned artifacts from those real source files
- the repo can synthesize Countdown-aligned fallback traces locally when a stronger trace source is unavailable
- the target Qwen model can complete a first local SFT checkpoint and a first local reset-aware RL checkpoint on a pilot subset
- evaluation can load trainer output roots directly even when the actual model lives inside `best-checkpoint` or `final-checkpoint`
- fallback SFT preparation can append retry-stage recovery examples and rebalance them explicitly
- the held-out hard slice can now be scored in both raw one-pass mode and reset-aware retry mode from the same checkpoint

What is not implemented yet:

- a strong paper-native expert-trace source that cleanly provides both productive and unproductive tagged traces for the reset-aware SFT stage
- a larger paper-style run with enough data and compute to produce nontrivial Countdown accuracy
- a properly scaled baseline-versus-reset-aware comparison on a larger hard Countdown slice
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
- an expanded fallback run with multiple synthetic solution variants kept the same held-out summary, but confirmed a key comparison:
  - raw one-pass generation stayed `0/4` valid
  - reset-aware evaluation stayed `4/4` valid with `clean_rate = 1.0`
- a walkthrough-enriched fallback run lowered local SFT loss further and changed the retry behavior qualitatively:
  - the model now emits explicit arithmetic steps after `<clean>`
  - the held-out slice still stays `0/4` correct and `4/4` valid
  - the current failure mode is arithmetic hallucination, where the narrated steps and target claim look plausible but the final expression still evaluates incorrectly
- the first walkthrough-based RL retry run exposed an execution constraint:
  - `max_new_tokens = 64` truncates longer retry walkthroughs and collapses validation validity to `0.0`
  - rerunning that RL stage with `max_new_tokens = 128` restores fully formed answers, but the held-out slice still remains `0/4` correct and `4/4` valid

Interpretation:

- the source wiring and runtime pipeline now work on real fetched assets
- the real-source fetch path now covers both Countdown prompts and reference traces, not only Countdown
- the current pilot is still too weak to claim meaningful paper-level performance
- the raw-vs-reset-aware gap is now directly measured on the held-out slice: reset logic helps validity, while one-pass generation still fails outright
- the main remaining baseline blocker is no longer formatting or reset behavior; it is arithmetic grounding and target-correctness after the clean step, plus larger compute/data
- the first extension module is now implemented separately from the baseline path, so future work can extend beyond one-shot cleaning without destabilizing the paper baseline

## Remaining Engineering Work

1. Scale the fetched reference-trace corpus beyond the tiny smoke slice and decide whether it is strong enough on its own or still needs a Countdown-native expert-trace supplement.
2. Strengthen the fallback recovery path further so the post-clean retry stage learns arithmetic that is actually consistent with the final expression, not just valid-looking structure.
3. Expand the paper-aligned Countdown source split beyond the tiny local slice.
4. Re-run the SFT stage on that larger aligned split.
5. Re-run the reset-aware RLOO stage on top of that checkpoint with a retry token budget that matches the trace format length.
6. Evaluate on hard Countdown examples and compare against the default baseline.
7. Extend the current multi-clean module toward selective retention and recall-aware memory.

## Remaining Non-Engineering Work

- choose the presentation slot in the shared schedule
- convert the technical material into the final five slides
- assign speaking order and rehearse timing for the 8-minute talk plus Q&A

## Recommended Talking Points Right Now

- the technical baseline is no longer just an idea; the core mechanics exist in code and are tested
- the repo now shows a direct held-out separation between raw one-pass failure and reset-aware valid retries
- the next meaningful milestone is stronger target-correct recovery after the clean step, ideally from a paper-native or larger Countdown-aligned trace source
- the model now reliably reaches well-formed post-clean traces on the held-out slice, so the remaining gap is arithmetic correctness rather than formatting
- the most important future direction remains stronger context management beyond one-shot reset

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
- verifier-grounded and contrastive synthetic recovery responses that explicitly state checked expression values
- hard-focused synthetic Countdown source generation for multiplication/division-heavy prompts
- failure-mined recovery trace generation from failed hard eval outputs
- automatic resolution of nested `best-checkpoint` and `final-checkpoint` model outputs
- a first bounded multi-step cleaning extension with clean-budget and clean-penalty support
- a selective-retention extension that carries explicit retained notes into retry prompts after clean
- a recall-aware memory extension with explicit memory writes, deterministic recall prompts, and a memory-aware clean loop
- a comparison utility for full reset, selective retention, and memory-aware clean trajectories
- retry-stage recovery augmentation for fallback SFT preparation
- a recovery-balance control for repeating retry-stage examples during fallback dataset prep
- an opt-in verifier gate that keeps only target-correct retry-stage recovery examples when Countdown metadata is available
- deterministic hard Countdown eval slicing for multiplication/division-heavy comparison runs
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
- evaluation outputs now include the verifier-computed expression value for each generated answer
- training runs emit local metrics and checkpoint outputs for inspection
- the repo can fetch and flatten the upstream Countdown train/eval datasets into local JSONL source files
- the repo can fetch positive reference traces and expand them into a paired reset-aware trace corpus
- the repo can prepare paper-aligned artifacts from those real source files
- the repo can synthesize Countdown-aligned fallback traces locally when a stronger trace source is unavailable
- the repo can force synthetic Countdown source generation onto hard multiplication/division-heavy prompts
- the repo can mine failed hard eval outputs into solver-verified recovery traces for the next SFT cycle
- the synthetic trace generator can emit walkthrough, verifier-grounded, and contrastive recovery styles for the same solved prompt
- raw one-pass and reset-aware evaluation summaries can be compared with a repeatable CLI utility
- the target Qwen model can complete local SFT checkpoints and reset-aware RL checkpoints on pilot subsets
- evaluation can load trainer output roots directly even when the actual model lives inside `best-checkpoint` or `final-checkpoint`
- fallback SFT preparation can append retry-stage recovery examples and rebalance them explicitly
- fallback SFT preparation can require retry-stage recovery examples to verify against Countdown numbers and target before appending them
- paper-aligned artifact prep now writes `countdown-test-hard.jsonl` beside the full eval set
- the held-out hard slice can now be scored in both raw one-pass mode and reset-aware retry mode from the same checkpoint
- verifier-grounded SFT ablations can now be compared against the expanded SFT and RLOO checkpoints
- contrastive recovery SFT ablations can now be compared against the expanded SFT and verifier-grounded checkpoints
- extension trajectories can now be compared in a shared reward table before scaling to larger train/eval runs

What is not implemented yet:

- a strong paper-native expert-trace source that cleanly provides both productive and unproductive tagged traces for the reset-aware SFT stage
- a larger paper-style run with enough data and compute to produce nontrivial Countdown accuracy
- a properly scaled raw-versus-reset-aware comparison on a larger hard Countdown slice
- scaled extension experiments that compare full reset, selective retention, and recall-aware memory

## Current Pilot Result

The latest local pilot completed on a CPU-only setup using fetched Countdown prompts, fetched reference traces, and an expanded solver-backed fallback trace set.

Observed pilot outcome:

- the source bundle used 32 fetched Countdown train prompts, 8 fetched Countdown eval prompts, and 64 fetched reference-trace rows expanded into 128 paired reset-aware traces
- the Countdown train source was expanded to 224 local train prompts, producing 868 synthetic Countdown trace records and 996 total hybrid trace records after merging reference traces
- prepared artifacts contained 1251 SFT train examples, 179 SFT validation examples, 168 Countdown train prompts, 56 Countdown validation prompts, and 8 held-out test prompts
- expanded SFT on `Qwen/Qwen2.5-0.5B` completed with `train_loss = 0.1813` and `eval_loss = 0.0477`
- raw one-pass generation from that SFT checkpoint scored `0/8` valid and `0/8` correct because the model emitted `<clean>` without a final answer
- reset-aware evaluation from the same SFT checkpoint scored `8/8` valid, `1/8` correct, `average_score = 0.225`, and `clean_rate = 1.0`
- a small reset-aware RLOO pass completed on top of that SFT checkpoint with 3 steps, 4 prompts per step, 4 responses per prompt, and `max_new_tokens = 128`
- RLOO found some correct sampled training rollouts, but the validation subset stayed `0/8` correct
- the RLOO checkpoint scored `7/8` valid, `1/8` correct, `average_score = 0.225`, and `clean_rate = 1.0` on the same held-out test prompts
- a verifier-grounded mixed recovery ablation completed at half an epoch with `2010` train examples and `288` validation examples; it scored `8/8` valid, `0/8` correct, and `average_score = 0.10`
- a balanced verifier-grounded recovery-only ablation completed with `1251` train examples, `179` validation examples, `train_loss = 0.1840`, and `eval_loss = 0.0545`
- that verification-only checkpoint scored `0/8` valid and `0/8` correct in raw one-pass decoding, but `8/8` valid, `1/8` correct, `average_score = 0.225`, and `clean_rate = 1.0` with reset-aware retry evaluation
- an all-style contrastive recovery ablation completed at half an epoch with `2770` train examples, `396` validation examples, `train_loss = 0.1324`, and `eval_loss = 0.0709`
- that contrastive/all checkpoint scored `0/8` valid and `0/8` correct in raw one-pass decoding, and `7/8` valid, `0/8` correct, `average_score = 0.10`, and `clean_rate = 1.0` with reset-aware retry evaluation
- a generated hard-slice check from the existing 8-example eval set produced `3` multiplication/division-heavy prompts
- on that hard slice, expanded SFT raw one-pass decoding scored `0/3` valid and `0/3` correct
- on that hard slice, expanded SFT reset-aware retry evaluation scored `3/3` valid, `0/3` correct, `average_score = 0.10`, and `clean_rate = 1.0`
- the next hard-focused local source build generated `64` hard Countdown prompts, `844` all-style solver-backed traces, and prepared `1329` SFT train examples plus `148` validation examples
- hard-focused SFT completed on that artifact set with `train_loss = 0.0750` and `eval_loss = 0.0963`
- hard-focused SFT raw one-pass decoding scored `0/3` valid and `0/3` correct on `countdown-test-hard.jsonl`
- hard-focused SFT reset-aware retry scored `3/3` valid, `0/3` correct, `average_score = 0.10`, and `clean_rate = 1.0`
- hard-focused reset-aware RLOO completed on top of that SFT checkpoint with 3 steps, 4 prompts per step, 4 responses per prompt, and `max_new_tokens = 128`
- hard-focused RLOO produced `final_loss = 0.0`, `best_validation_accuracy = 0.0`, and `best_validation_average_score = 0.10`, which means the sampled rollouts produced no positive correctness advantage
- the hard-focused RLOO checkpoint scored `0/3` valid and `0/3` correct in raw one-pass decoding, then `3/3` valid and `0/3` correct with reset-aware retry
- failure mining over the hard-focused SFT eval output produced `15` solver-verified recovery records from the `3` failed hard examples
- a combined plus-mined local artifact set prepared successfully with `1356` SFT train examples and `151` validation examples

Interpretation:

- the source wiring and runtime pipeline now work on real fetched assets plus local Countdown-aligned trace expansion
- the raw-vs-reset-aware gap is now directly measured on an 8-example held-out slice: reset logic converts raw invalid outputs into valid retry answers
- the best current local checkpoint is the expanded SFT checkpoint, not the small RLOO checkpoint, because it keeps the same held-out correctness while preserving full validity
- the balanced verifier-grounded ablation ties the expanded SFT checkpoint on held-out correctness and validity, but does not improve beyond it
- the contrastive/all ablation also regresses below the expanded SFT checkpoint, so the next data step should improve trace quality rather than simply increasing reset-style volume
- the current pilot is still too weak to claim strong target-model performance
- the hard-focused SFT and RLOO results confirm that reset improves answer format on harder prompts, but has not yet produced target-correct hard arithmetic
- the hard-focused RLOO loss stayed at zero because the policy saw no correctness-reward variation on those sampled hard prompts
- the mined recovery records are useful for the next training cycle, but they must be evaluated against a fresh hard holdout because they were derived from the current hard failures
- the main remaining baseline blocker is arithmetic grounding: the model often writes plausible step-by-step claims, but the verifier-computed expression value does not match the target
- the first extension module remains separate from the baseline path, so future multi-clean work can proceed without destabilizing the one-shot baseline
- the second extension module now supports explicit retained notes after clean, while still avoiding full scratchpad carryover
- the recall-aware memory module now provides a runnable external-memory clean-loop baseline for future extension experiments

## Remaining Engineering Work

1. Train on the plus-mined recovery artifact set only after creating a fresh hard holdout for honest evaluation.
2. Add or fetch a stronger Countdown-native expert-trace source with verified target-correct hard recoveries.
3. Scale the fetched reference-trace corpus and Countdown split beyond the current local CPU pilot.
4. Re-run SFT and reset-aware RLOO after the trace source can produce positive correctness rewards on hard prompts.
5. Run a larger raw-versus-reset-aware comparison using a broader hard Countdown slice.
6. Use the extension comparison utility to scale full reset, selective retention, and memory-aware clean-loop comparisons.

## Remaining Non-Engineering Work

- choose the presentation slot in the shared schedule
- convert the technical material into the final five slides
- assign speaking order and rehearse timing for the 8-minute talk plus Q&A

## Recommended Talking Points Right Now

- the technical baseline is no longer just an idea; the core mechanics exist in code and are tested
- the repo now shows a direct held-out separation between raw one-pass failure and reset-aware valid retries
- the next meaningful milestone is stronger target-correct recovery after the clean step, ideally from a paper-native or larger Countdown-aligned trace source
- the expanded SFT and balanced verifier-grounded checkpoints both reach one target-correct held-out retry, while the small RLOO pass does not improve that held-out result
- the contrastive/all ablation did not improve the held-out result, which narrows the next technical bet to better trace quality rather than more recovery trace volume
- the model now reliably reaches mostly well-formed post-clean traces on the held-out slice, so the remaining gap is arithmetic correctness rather than reset formatting
- the most important future direction remains stronger context management beyond one-shot reset, now including selective retention and memory-aware clean loops

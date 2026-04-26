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
- arithmetic-grounded synthetic recovery traces with substep arithmetic, rejected hypothesis, number-budget audit, and final-value reconciliation, randomized by `rng_seed`
- a deep-verify pipeline filter that re-checks every inline `a op b = c` claim before mined or synthetic recoveries enter SFT
- a multi-clean evaluation decoder (`eval_runtime --max-clean-tries N`) with verifier-aware first-correct early stop
- `score_when_cleaned` and `clean_rate` summary metrics that align reset-aware reporting with paper Figure 6
- a contamination guard on `failure_recovery_traces` that aborts mining when source IDs overlap a held-out slice
- a Figure 7-style qualitative best/worst sample exporter (`scripts/export_qualitative_samples.py`)
- `--scale {pilot,paper}` presets on `paper_sources` and `prepare_paper_artifacts` (paper preset matches Section 4.1 settings)
- an end-to-end `scripts/replicate_paper.sh` pipeline with a `--dry-run` smoke mode and a `LTR_REPLICATE_DRY_RUN`-gated subprocess test
- a controller-comparison runner (`scripts/run_extension_comparison.py`) that scores `full_reset`, `selective_retention`, and `memory` on identical multi-clean eval segments
- a paper-claim traceability document mapping every `main.pdf` Section 3/4 claim to its file and test
- a paper-replication runbook covering hardware, wall-clock, expected hard-Countdown band, and verification checks
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
- the synthetic trace generator can emit walkthrough, verifier-grounded, contrastive, and arithmetic-grounded (`grounded`) recovery styles for the same solved prompt
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
- a fresh generated hard holdout was created with `32/32` multiplication/division-heavy test examples
- plus-mined SFT completed on the fresh-holdout artifact set with `train_loss = 0.0575` and `eval_loss = 0.1096`
- plus-mined SFT raw one-pass decoding on the fresh hard holdout scored `0/32` valid and `0/32` correct
- plus-mined SFT reset-aware retry on the fresh hard holdout scored `29/32` valid, `1/32` correct, `average_score = 0.13125`, and `clean_rate = 1.0`
- arithmetic-grounded recovery (R16) shipped in code: `build_grounded_recovery_response`, `grounded` in `RecoveryStyle` / CLIs, and `pipeline._deep_verify_recovery_text` when `--require-recovery-target-correct` is set (132 unit tests green)
- a grounded-plus-mined artifact set was prepared from the same hard-focused train source as the plus-mined baseline plus 15 non-contaminated mined recoveries (`1736` train / `193` validation SFT examples in `tmp/paper-artifacts-grounded-26apr/`)
- SFT on `Qwen/Qwen2.5-0.5B` over that set completed to `tmp/paper-runs/sft-grounded-26apr/` with `train_loss ≈ 0.0903`
- reset-aware eval on the same 32-example fresh hard holdout scored `30/32` valid and `0/32` correct at `max_new_tokens = 128`; re-eval at `384` tokens was unchanged (`0/32` correct)
- raw one-pass on that holdout stayed `0/32` valid and `0/32` correct
- a second 32-example hard slice (`seed = 131`) scored `28/32` valid and `0/32` correct under reset-aware retry, ruling out a one-off fluke on the original holdout
- a grounded-only ablation (`569` train / `64` validation examples) trained to `tmp/paper-runs/sft-grounded-only-26apr/`; at `max_new_tokens = 384` reset-aware eval reached `31/32` valid and `0/32` correct — the model completes the grounded template but fabricates substep arithmetic (see `docs/grounded-recovery-results-2026-04-26.md`)
- the plus-mined baseline was re-evaluated at `max_new_tokens = 384` and stayed `1/32` correct, so the token budget is not the primary limiter for the `1/32` bar
- the grounded SFT checkpoint (`tmp/paper-runs/sft-grounded-26apr/`) was re-evaluated at `max_new_tokens = 384` with `--max-clean-tries 3` on both holdouts: fresh-32 stayed `0/32` correct with `clean_rate = 1.0` and `score_when_cleaned = 0.094`, and the `seed = 131` slice stayed `0/32` correct with `clean_rate = 1.0` and `score_when_cleaned = 0.088`; the multi-clean decoder alone does not raise the `1/32` bar at the 0.5B scale
- a controller comparison on the same fresh-32 multi-clean segments found `full_reset`, `selective_retention`, and `memory` all tied at `mean_adjusted_reward ≈ 0.05` because the underlying segments themselves never reach the target — a controller swap on identical generations cannot beat the substep-fabrication ceiling
- a third hard slice (`seed = 219`, `tmp/paper-artifacts-grounded-26apr-seed219/countdown-test-hard.jsonl`) was generated with renamespaced `seed219-synthetic:N` source IDs to keep it disjoint from fresh-32 and `seed = 131`; its raw baseline eval against the existing grounded SFT scored `0/32` correct, `0/32` valid, and `clean_rate = 1.0`, and the contamination-guarded mining pass produced `62` solver-verified recovery traces written to `tmp/paper-assets-grounded-26apr-seed219/mined-recoveries-grounded.jsonl`
- those recoveries were concatenated into the grounded SFT corpus, artifacts were re-prepared under `tmp/paper-artifacts-grounded-26apr-seed219mine/`, and one epoch of SFT on `Qwen/Qwen2.5-0.5B` completed to `tmp/paper-runs/sft-grounded-26apr-seed219mine/` with `train_loss ≈ 0.0695`, `eval_loss ≈ 0.103`
- post-retrain multi-clean evaluation (`max_new_tokens = 384`, `--max-clean-tries 3`) on the same two holdouts: fresh-32 scored `0/32` correct, `31/32` valid, `clean_rate = 1.0`, `score_when_cleaned ≈ 0.097`; `seed = 131` scored `0/32` correct, `32/32` valid, `clean_rate = 1.0`, `score_when_cleaned = 0.10` (outputs in `tmp/paper-eval/sft-grounded-26apr-seed219mine-fresh32-multiclean3/` and `tmp/paper-eval/sft-grounded-26apr-seed219mine-seed131-multiclean3/`)
- **Phase B decision gate (B4):** the plan’s pass condition was reset-aware target correctness **strictly greater than** `1/32` on at least one of {fresh-32, `seed = 131`} after B1 or B2; both holdouts remained **`0/32` correct** after B2, so the gate **did not pass**. Validity improved versus the pre-retrain multi-clean grounded checkpoint (`30/32` and `28/32` valid respectively on the same metric), but arithmetic target correctness did not move — document as a **negative finding** for this lever set at 0.5B, not as a silent retry loop
- **Verifier-feedback ablation (post-B4):** the same checkpoint (`tmp/paper-runs/sft-grounded-26apr-seed219mine`) was re-evaluated with `--max-clean-tries 3 --verifier-feedback` (decode-time hints after each `<clean>`). Fresh-32 stayed **`0/32` correct**, `31/32` valid, `score_when_cleaned ≈ 0.097` (`tmp/paper-eval/sft-seed219mine-fresh32-feedback3/`). `seed = 131` stayed **`0/32` correct** with **`30/32` valid** (down from `32/32` without the flag), `score_when_cleaned ≈ 0.094` (`tmp/paper-eval/sft-seed219mine-seed131-feedback3/`). So explicit verifier text in the retry prompt **did not** unlock target correctness at 0.5B on these slices; the next levers remain larger models (`scripts/replicate_paper.sh`), RLOO with verifier-shaped reward, or solver-in-the-loop product modes outside strict LM accuracy

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
- the fresh hard holdout shows a small target-correct improvement, but most retry responses still state false verified equations, so arithmetic verification is still the main blocker
- the main remaining baseline blocker is arithmetic grounding: the model often writes plausible step-by-step claims, but the verifier-computed expression value does not match the target
- the grounded recovery template increases structural supervision (substeps, rejected hypothesis, budget, final check) but at Qwen2.5-0.5B the model still copies the shape while lying on intermediate `Compute` lines; deep-verify cleans training data but cannot fix inference-time fabrication — the next bet is step-level verifier reward or a larger model, not another passive template tweak alone
- the first extension module remains separate from the baseline path, so future multi-clean work can proceed without destabilizing the one-shot baseline
- the second extension module now supports explicit retained notes after clean, while still avoiding full scratchpad carryover
- the recall-aware memory module now provides a runnable external-memory clean-loop baseline for future extension experiments

## Remaining Engineering Work

1. Improve recovery supervision with step-level verifier reward (or similar) so intermediate `Compute` claims cannot be fabricated without penalty; passive templates alone did not beat the `1/32` hard-holdout bar at 0.5B.
2. Add or fetch a stronger Countdown-native expert-trace source with verified target-correct hard recoveries.
3. Scale the fetched reference-trace corpus and Countdown split beyond the current local CPU pilot.
4. Re-run reset-aware RLOO after the SFT checkpoint produces more than sparse target-correct hard retries.
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

## Paper Alignment

What is now true vs the paper (`main.pdf` Section 3 and Section 4):

- Every Section 3 mechanism the paper cites — `<clean>`-aware SFT trace curation (Section 3.2.1 / Figure 3), one-shot context reset (Section 3.2.2 / Figure 4), reward propagation through both segments (Section 3.2.3 / Eq. 4-6), the multi-clean controller, and the selective-retention and memory extensions referenced in the Discussion — is implemented in code and exercised by tests. Each row in `docs/paper-claims-traceability.md` maps one paper claim to one file plus one or more tests.
- Every Section 4 reporting surface is now in tree as well: Figure 6's `score_when_cleaned` and `clean_rate` are reported by `evaluate_countdown_outputs`; Figure 7's qualitative best/worst examples are emitted by `scripts/export_qualitative_samples.py`; the verifier-aware multi-clean decoder and the contamination-guarded failure miner that Section 4.3 relies on are wired into `eval_runtime` (`--max-clean-tries`) and `failure_recovery_traces` (`--exclude-source-ids`) respectively.
- Section 4's headline 13.61% → 36.94% hard-Countdown jump is reported by the paper on a 1B-class target model. We do not have GPU compute to run that target locally. Instead we ship the path: `--scale {pilot,paper}` presets on `paper_sources` and `prepare_paper_artifacts` flip every dataset cap to the Section 4.1 settings, `scripts/replicate_paper.sh` drives the full ten-step pipeline (sources → SFT → mining → re-SFT → RLOO → multi-clean eval → extension comparison → qualitative export), `--dry-run` validates that every CLI parses on Qwen 2.5 0.5B without spending GPU minutes, and `docs/paper-replication.md` documents hardware (≥1× A100 40GB), wall-clock, and the verification checks needed to claim a faithful replication.
- The honest limit at 0.5B is documented: with multi-clean retries and verifier rerank wired in, both fresh-32 and `seed = 131` evals stayed at `0/32` correct after B1 **and** after B2 (seed-219 mined recoveries + re-SFT); `clean_rate = 1.0` and `score_when_cleaned ≈ 0.09–0.10` throughout, with validity rising to `31/32` and `32/32` valid only after B2. A follow-on decode-only ablation that **injected verifier arithmetic text into every retry prompt** (`--verifier-feedback`) still scored **`0/32` correct** on both holdouts, so the failure mode is not “missing feedback in the prompt” alone. The controller comparison on identical segments tied all three policies at `mean_adjusted_reward ≈ 0.05`. The paper's hard-Countdown claim therefore remains an explicitly out-of-scope replication target on local CPU/MPS, and the correct path to validate it is the documented 1B+ replication run, not another 0.5B template iteration.

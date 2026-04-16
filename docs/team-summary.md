# Team Summary

## Done

- Repository is set up with private GitHub remote and CI.
- Core baseline mechanics are implemented and tested:
  - trace curation for `<clean>`-aware examples
  - one-shot reset flow `y0 -> optional <clean> -> y1`
  - reset-aware RLOO reward and advantage utilities
- Countdown-aligned fallback support now exists:
  - deterministic Countdown solver
  - synthetic positive/negative trace generation from local Countdown prompts
  - multiple synthetic solution variants per prompt
  - step-by-step arithmetic walkthrough traces after reset
  - verifier-grounded recovery traces that state the checked expression value
  - contrastive recovery traces that reject the failed expression before giving the verified answer
  - hard-focused source generation for multiplication/division-heavy prompts
- Data/prompt preparation is in place:
  - trace and Countdown loaders
  - prompt assembly and split/export tooling
- Local runtime entrypoints exist for:
  - SFT
  - reset-aware RLOO
  - clean-aware evaluation
- Evaluation records now include the verifier-computed expression value, which makes arithmetic failures easier to diagnose.
- Raw one-pass and reset-aware evaluation summaries can now be compared with a repeatable CLI utility.
- Real-source Countdown adapters now exist:
  - upstream train/eval fetch and flattening
  - reference trace fetch plus paired trace bootstrap
  - paper-aligned artifact preparation from separate train/eval files
  - automatic `countdown-test-hard.jsonl` export for multiplication/division-heavy evals
  - a tiny live fetch -> bootstrap -> prep smoke run
- Trainer outputs can now be evaluated directly from the run root because nested checkpoints resolve automatically.
- Fallback SFT prep can now:
  - append retry-stage recovery examples
  - rebalance those recovery examples with a repeat factor
  - require retry-stage recovery examples to verify against Countdown numbers and target
- First extension work is already in code:
  - bounded multi-step cleaning
  - per-clean penalty support
  - explicit selective retention with `<retain>...</retain>` notes carried into retry prompts
  - recall-aware memory with `<memory>...</memory>` writes, deterministic recall prompts, and a memory-aware clean loop
  - extension comparison tables for full reset, selective retention, and memory-aware clean loops
- The latest expanded local run completed:
  - expanded SFT reached `8/8` valid and `1/8` correct on the held-out reset-aware slice
  - raw one-pass decoding stayed `0/8` valid
  - a small RLOO pass completed but did not improve held-out correctness
- Verifier-grounded recovery ablations completed:
  - mixed verifier-grounded recovery reached `8/8` valid but `0/8` correct
  - balanced verifier-grounded recovery tied the best SFT result at `8/8` valid and `1/8` correct
- Contrastive/all recovery ablation completed:
  - half-epoch SFT used `2770` train examples and `396` validation examples
  - reset-aware evaluation reached `7/8` valid but `0/8` correct
  - raw one-pass decoding remained `0/8` valid and `0/8` correct
- Hard-slice check completed on the generated `countdown-test-hard.jsonl` subset:
  - existing 8-example eval set produced `3` hard prompts
  - raw one-pass decoding scored `0/3` valid and `0/3` correct
  - reset-aware retry scored `3/3` valid but `0/3` correct
- Hard-focused artifact set and training run completed locally:
  - `64` generated hard Countdown prompts
  - `844` solver-backed hard traces
  - `1329` SFT train examples and `148` validation examples
  - hard-focused SFT finished with `train_loss = 0.0750` and `eval_loss = 0.0963`
  - hard-focused SFT scored `0/3` valid raw, then `3/3` valid and `0/3` correct with reset-aware retry
  - hard-focused RLOO finished with `final_loss = 0.0` and `best_validation_accuracy = 0.0`
  - hard-focused RLOO also scored `3/3` valid and `0/3` correct with reset-aware retry
- Failure-mined recovery traces are now available:
  - `15` solver-verified recovery records mined from the `3` failed hard eval examples
  - plus-mined artifact set prepared with `1356` SFT train examples and `151` validation examples
  - these are for the next training cycle, not for reporting on the same mined hard examples
- Fresh hard-holdout plus-mined SFT completed:
  - fresh hard holdout has `32/32` hard examples
  - SFT finished with `train_loss = 0.0575` and `eval_loss = 0.1096`
  - raw one-pass decoding stayed `0/32` valid and `0/32` correct
  - reset-aware retry reached `29/32` valid and `1/32` correct
- A runnable baseline demo is available through:
  - `PYTHONPATH=src python3 -m learning_to_reset`

## To Do

- Improve recovery traces so the model actually checks arithmetic values instead of copying false “verified” equations.
- Add or build a stronger Countdown-native expert-trace source with verified target-correct hard recoveries.
- Re-run reset-aware RLOO after SFT produces more than sparse correctness reward on hard prompts.
- Scale the local CPU pilot into a larger target-model run.
- Run raw-versus-reset-aware comparison on a larger hard Countdown slice.
- Use the extension comparison utility to scale full reset, selective retention, and memory-aware clean-loop comparisons.

## Next Plan

1. Add stronger arithmetic-checking recovery supervision.
2. Train the next SFT checkpoint on the improved verified recovery traces.
3. Run reset-aware RLOO only after sampled hard rollouts produce a stronger correctness signal.
4. Compare the next checkpoint against raw generation, expanded SFT, verifier-grounded SFT, and the RLOO checkpoints.
5. Continue the extension track by comparing full reset, selective retention, and memory-aware clean loops after the one-shot baseline is stronger.

## Latest Signal

- Raw one-pass generation is still `0/3` valid on the held-out hard slice.
- The best expanded SFT checkpoint is `8/8` valid, `1/8` correct, with `clean_rate = 1.0` under reset-aware evaluation.
- The balanced verifier-grounded checkpoint ties that best SFT result and also keeps `8/8` validity.
- The mixed verifier-grounded checkpoint regressed to `0/8` correct, so adding more recovery-style traces by volume is not enough.
- The contrastive/all checkpoint also regressed to `7/8` valid and `0/8` correct, so the next bet is trace quality, not trace volume.
- The small RLOO pass is `7/8` valid and `1/8` correct on the same held-out slice, so it does not beat SFT yet.
- The hard-focused SFT and hard-focused RLOO runs both reached `3/3` valid but `0/3` correct on the hard slice.
- Failure mining now gives targeted recovery data, but the next metric needs a fresh hard holdout to avoid leakage.
- On the fresh hard holdout, plus-mined SFT reached `29/32` valid and `1/32` correct with reset-aware retry.
- Current blocker: arithmetic grounding on unseen hard Countdown prompts, not reset formatting.

## Non-Engineering Leftovers

- Pick the team presentation slot.
- Convert the technical material into the final five slides.
- Rehearse speaking order, timing, and Q&A.

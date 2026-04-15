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
- Data/prompt preparation is in place:
  - trace and Countdown loaders
  - prompt assembly and split/export tooling
- Local runtime entrypoints exist for:
  - SFT
  - reset-aware RLOO
  - clean-aware evaluation
- Evaluation records now include the verifier-computed expression value, which makes arithmetic failures easier to diagnose.
- Real-source Countdown adapters now exist:
  - upstream train/eval fetch and flattening
  - reference trace fetch plus paired trace bootstrap
  - paper-aligned artifact preparation from separate train/eval files
  - a tiny live fetch -> bootstrap -> prep smoke run
- Trainer outputs can now be evaluated directly from the run root because nested checkpoints resolve automatically.
- Fallback SFT prep can now:
  - append retry-stage recovery examples
  - rebalance those recovery examples with a repeat factor
- First extension work is already in code:
  - bounded multi-step cleaning
  - per-clean penalty support
- The latest expanded local run completed:
  - expanded SFT reached `8/8` valid and `1/8` correct on the held-out reset-aware slice
  - raw one-pass decoding stayed `0/8` valid
  - a small RLOO pass completed but did not improve held-out correctness
- A runnable baseline demo is available through:
  - `PYTHONPATH=src python3 -m learning_to_reset`

## To Do

- Regenerate the hybrid synthetic traces with verifier-grounded recovery style enabled.
- Add or build a stronger Countdown-native expert-trace source if the current hybrid trace source remains too weak.
- Re-run SFT after the arithmetic-grounding improvement.
- Re-run reset-aware RLOO after SFT produces stronger target-correct retries.
- Scale the local CPU pilot into a larger target-model run.
- Extend the multi-clean branch toward selective retention and recall-aware memory.

## Next Plan

1. Rebuild the expanded hybrid trace set with both walkthrough and verifier-grounded recovery examples.
2. Re-run SFT on the improved aligned trace set.
3. Add a stronger Countdown-native trace source if the verifier-grounded synthetic path is still too weak.
4. Run reset-aware RLOO only after the SFT checkpoint improves target-correct retries.
5. Keep comparing raw one-pass mode versus reset-aware mode on the same held-out slice.

## Latest Signal

- Raw one-pass generation is still `0/8` valid on the held-out hard slice.
- The best expanded SFT checkpoint is `8/8` valid, `1/8` correct, with `clean_rate = 1.0` under reset-aware evaluation.
- The small RLOO pass is `7/8` valid and `1/8` correct on the same held-out slice, so it does not beat SFT yet.
- Current blocker: arithmetic grounding on unseen Countdown prompts, not reset formatting.

## Non-Engineering Leftovers

- Pick the team presentation slot.
- Convert the technical material into the final five slides.
- Rehearse speaking order, timing, and Q&A.

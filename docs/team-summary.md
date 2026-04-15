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
- Data/prompt preparation is in place:
  - trace and Countdown loaders
  - prompt assembly and split/export tooling
- Local runtime entrypoints exist for:
  - SFT
  - reset-aware RLOO
  - clean-aware evaluation
- Real-source Countdown adapters now exist:
  - upstream train/eval fetch and flattening
  - paper-aligned artifact preparation from separate train/eval files
- Trainer outputs can now be evaluated directly from the run root because nested checkpoints resolve automatically.
- Fallback SFT prep can now:
  - append retry-stage recovery examples
  - rebalance those recovery examples with a repeat factor
- First extension work is already in code:
  - bounded multi-step cleaning
  - per-clean penalty support
- A runnable baseline demo is available through:
  - `PYTHONPATH=src python3 -m learning_to_reset`

## To Do

- Replace the fallback trace source with a stronger paper-native expert-trace source.
- Add better recovery supervision so clean-triggered retries become arithmetic-consistent and target-correct, not just valid-looking.
- Expand the local Countdown source split beyond the tiny pilot slice.
- Re-run SFT on the larger paper-aligned split.
- Re-run the reset-aware RLOO stage on top of that checkpoint with the right retry token budget.
- Evaluate baseline vs reset-aware results on a larger hard Countdown slice.
- Extend the multi-clean branch toward selective retention and recall-aware memory.

## Next Plan

1. Strengthen the trace source and expand the Countdown slice.
2. Push recovery quality from valid-looking retries toward arithmetic-consistent, correct answers.
3. Run SFT on the larger aligned trace set.
4. Run reset-aware RLOO on the resulting checkpoint with a retry budget that fits the trace format.
5. Compare hard-example performance in raw one-pass mode versus reset-aware mode.

## Latest Signal

- Raw one-pass generation is still `0/4` valid on the held-out hard slice.
- Reset-aware evaluation is consistently `4/4` valid with `clean_rate = 1.0` on that same slice.
- The newest walkthrough traces improved the qualitative retry behavior, but the held-out slice is still `0/4` correct.
- Current blocker: arithmetic grounding on unseen Countdown prompts, not reset formatting.

## Non-Engineering Leftovers

- Pick the team presentation slot.
- Convert the technical material into the final five slides.
- Rehearse speaking order, timing, and Q&A.

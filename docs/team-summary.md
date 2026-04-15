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
- First extension work is already in code:
  - bounded multi-step cleaning
  - per-clean penalty support
- A runnable baseline demo is available through:
  - `PYTHONPATH=src python3 -m learning_to_reset`

## To Do

- Replace the fallback trace source with a stronger paper-native expert-trace source.
- Add better recovery supervision so clean-triggered retries end in valid answers.
- Re-run SFT on a larger paper-aligned split.
- Re-run the reset-aware RLOO stage on top of that checkpoint.
- Evaluate baseline vs reset-aware results on a larger hard Countdown slice.
- Extend the multi-clean branch toward selective retention and recall-aware memory.

## Next Plan

1. Strengthen the trace source.
2. Add retry-recovery training signal.
3. Run SFT on the larger aligned trace set.
4. Run reset-aware RLOO on the resulting checkpoint.
5. Compare hard-example performance, clean usage, and recovery quality.

## Non-Engineering Leftovers

- Pick the team presentation slot.
- Convert the technical material into the final five slides.
- Rehearse speaking order, timing, and Q&A.

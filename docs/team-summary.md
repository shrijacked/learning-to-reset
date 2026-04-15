# Team Summary

## Done

- Repository is set up with private GitHub remote and CI.
- Core baseline mechanics are implemented and tested:
  - trace curation for `<clean>`-aware examples
  - one-shot reset flow `y0 -> optional <clean> -> y1`
  - reset-aware RLOO reward and advantage utilities
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
- A runnable baseline demo is available through:
  - `PYTHONPATH=src python3 -m learning_to_reset`

## To Do

- Replace the provisional positive-trace source with a stronger expert-trace source.
- Re-run SFT on a larger paper-aligned split.
- Re-run the reset-aware RLOO stage on top of that checkpoint.
- Evaluate baseline vs reset-aware results on a larger hard Countdown slice.
- Begin extension work only after the baseline run is stable.

## Next Plan

1. Wire in the real datasets.
2. Strengthen the trace source.
3. Run SFT on the larger trace set.
4. Run reset-aware RLOO on the resulting checkpoint.
5. Compare hard-example performance and clean usage.

## Non-Engineering Leftovers

- Pick the team presentation slot.
- Convert the technical material into the final five slides.
- Rehearse speaking order, timing, and Q&A.

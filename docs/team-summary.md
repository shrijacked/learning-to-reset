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
- A runnable baseline demo is available through:
  - `PYTHONPATH=src python3 -m learning_to_reset`

## To Do

- Point the pipeline at the real expert-trace and Countdown files.
- Run the first SFT checkpoint on the target model.
- Run the reset-aware RLOO stage on top of that checkpoint.
- Evaluate baseline vs reset-aware results on hard Countdown examples.
- Begin extension work only after the baseline run is stable.

## Next Plan

1. Wire in the real datasets.
2. Run SFT on curated traces.
3. Run reset-aware RLOO on the SFT checkpoint.
4. Compare hard-example performance and clean usage.
5. Move to multi-step cleaning and selective retention after the baseline is stable.

## Non-Engineering Leftovers

- Pick the team presentation slot.
- Convert the technical material into the final five slides.
- Rehearse speaking order, timing, and Q&A.

# Team Summary

## Done

- Repository is set up with private GitHub remote and CI.
- Core baseline mechanics are implemented and tested:
  - trace curation for `<clean>`-aware examples
  - one-shot reset flow `y0 -> optional <clean> -> y1`
  - reset-aware RLOO reward and advantage utilities
- A runnable baseline demo is available through:
  - `PYTHONPATH=src python3 -m learning_to_reset`

## To Do

- Add expert-trace dataset ingestion.
- Add Countdown dataset loaders and evaluation harness.
- Build prompt assembly and batching code.
- Implement the SFT training loop.
- Implement the reset-aware RLOO training loop.
- Run the first baseline vs reset-aware evaluations.

## Next Plan

1. Wire in the datasets and prompt pipeline.
2. Run the first SFT path on curated traces.
3. Add the RLOO training stage on top of the SFT checkpoint.
4. Evaluate on hard Countdown examples.
5. Move to multi-step cleaning and selective retention after the baseline is stable.

## Non-Engineering Leftovers

- Pick the team presentation slot.
- Convert the technical material into the final five slides.
- Rehearse speaking order, timing, and Q&A.

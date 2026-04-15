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
- Raw one-pass and reset-aware evaluation summaries can now be compared with a repeatable CLI utility.
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
- Verifier-grounded recovery ablations completed:
  - mixed verifier-grounded recovery reached `8/8` valid but `0/8` correct
  - balanced verifier-grounded recovery tied the best SFT result at `8/8` valid and `1/8` correct
- A runnable baseline demo is available through:
  - `PYTHONPATH=src python3 -m learning_to_reset`

## To Do

- Add or build a stronger Countdown-native expert-trace source because the current hybrid source is valid but still weak on arithmetic correctness.
- Re-run SFT after the trace source improves target-correct post-clean retries.
- Re-run reset-aware RLOO only after SFT improves beyond the current `1/8` held-out correctness result.
- Scale the local CPU pilot into a larger target-model run.
- Run raw-versus-reset-aware comparison on a larger hard Countdown slice.
- Extend the multi-clean branch toward selective retention and recall-aware memory.

## Next Plan

1. Improve the Countdown trace source so retries are target-correct more often, not just well-formed.
2. Re-run SFT on the improved trace set and keep the same held-out comparison gate.
3. Compare the improved SFT checkpoint against raw generation, the current expanded SFT checkpoint, and the RLOO checkpoint.
4. Run reset-aware RLOO only if the improved SFT checkpoint beats the current `1/8` held-out correctness result.
5. Continue the extension track with selective retention and recall-aware memory after the one-shot baseline is stronger.

## Latest Signal

- Raw one-pass generation is still `0/8` valid on the held-out hard slice.
- The best expanded SFT checkpoint is `8/8` valid, `1/8` correct, with `clean_rate = 1.0` under reset-aware evaluation.
- The balanced verifier-grounded checkpoint ties that best SFT result and also keeps `8/8` validity.
- The mixed verifier-grounded checkpoint regressed to `0/8` correct, so adding more recovery-style traces by volume is not enough.
- The small RLOO pass is `7/8` valid and `1/8` correct on the same held-out slice, so it does not beat SFT yet.
- Current blocker: arithmetic grounding on unseen Countdown prompts, not reset formatting.

## Non-Engineering Leftovers

- Pick the team presentation slot.
- Convert the technical material into the final five slides.
- Rehearse speaking order, timing, and Q&A.

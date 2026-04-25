# Progress Summary Since 28 March 2026

This note covers the repository work completed from 28 March 2026 through 16 April 2026. It is meant to help teammates quickly understand what now exists in code, what the latest results say, and what the next work block should be.

## Snapshot

Since 28 March, the repository has moved from planning notes into a runnable research codebase. The project now supports paper-aligned source fetching, artifact preparation, supervised fine-tuning, reset-aware evaluation, reset-aware RLOO training, hard-problem slicing, failure-mined recovery supervision, and three extension paths: multi-clean control, selective retention, and recall-aware memory.

The main empirical signal so far is consistent: the reset path improves answer validity a lot on hard Countdown prompts, but arithmetic correctness is still the bottleneck.

## Timeline

| Date | What changed |
| --- | --- |
| 28 March 2026 | Added the first shareable project docs, including the refreshed `README` and the code walkthrough. |
| 15 April 2026 | Built the main baseline pipeline: data and prompt utilities, split/export tooling, Countdown verification, SFT and RLOO runtimes, paper-source adapters, solver-backed fallback traces, retry recovery supervision, and the first multi-clean extension block. |
| 16 April 2026 | Expanded the source and diagnostics layer, added verifier-grounded and contrastive recovery variants, added selective retention and memory extensions, added hard-slice evaluation and hard-focused training, mined hard failures into recovery traces, and recorded the fresh hard-holdout result. |

## What Landed In Code

### 1. End-to-end baseline pipeline

- Data loaders, prompt builders, split helpers, and JSONL export are implemented.
- Paper-source adapters now fetch and prepare Countdown and reference-trace assets.
- Local runtimes exist for:
  - SFT
  - clean-aware evaluation
  - reset-aware RLOO
- Evaluation outputs now include verifier-computed expression values for debugging wrong-target answers.
- Raw one-pass and reset-aware runs can now be compared with a dedicated result-comparison utility.

### 2. Stronger Countdown-specific supervision

- Added a deterministic Countdown solver and answer verifier.
- Added synthetic fallback trace generation from Countdown prompts.
- Added multiple recovery styles, including walkthrough, verifier-grounded, and contrastive recovery traces.
- Added deterministic hard-slice extraction for multiplication/division-heavy evaluation.
- Added hard-focused prompt generation for stronger arithmetic training data.
- Added failure mining that converts hard eval failures into solver-verified recovery traces for the next SFT cycle.

### 3. Extension track is already implemented

- Bounded multi-clean support now exists with a clean budget and per-clean penalty.
- Selective retention is implemented through explicit retained notes that survive a clean reset.
- Recall-aware memory is implemented through explicit memory writes, deterministic recall, and a memory-aware retry flow.
- Extension trajectories can now be compared in a shared JSON/Markdown result table.

### 4. Repo quality and reproducibility improvements

- The main docs were expanded so the codebase is easier to navigate.
- GitHub CI is set up for the test suite.
- The latest known full test run passed `123` tests.
- Model/checkpoint resolution and cached snapshot selection were made deterministic.

## Best Current Results

### Core baseline signal

- Expanded SFT on the initial held-out slice stayed `0/8` valid and `0/8` correct in raw one-pass decoding.
- The same checkpoint reached `8/8` valid and `1/8` correct under reset-aware retry evaluation.
- A balanced verifier-grounded ablation matched that best baseline signal rather than clearly beating it.
- A small reset-aware RLOO pass completed successfully, but did not improve held-out correctness beyond the strongest SFT checkpoint.

### Hard Countdown signal

- On the original hard slice, hard-focused SFT and hard-focused RLOO both improved validity after reset, but neither produced target-correct hard answers.
- Failure mining over those hard failures produced `15` solver-verified recovery records for the next training round.
- On a fresh `32`-example hard holdout, the plus-mined SFT checkpoint stayed `0/32` valid and `0/32` correct in raw one-pass decoding.
- On that same fresh hard holdout, reset-aware retry reached `29/32` valid and `1/32` correct.

## What These Results Mean

- The reset-control mechanism is working as a behavioral tool.
- The training and evaluation loop now runs end to end on the intended setup.
- The project now has a concrete hard-case recovery-data pipeline instead of only generic synthetic supervision.
- The main blocker is no longer answer formatting after reset.
- The main blocker is arithmetic grounding on unseen hard Countdown prompts.

## What Still Needs To Be Done

1. Improve recovery supervision so the model actually checks arithmetic values instead of copying false "verified" equations.
2. Add or build a stronger Countdown-native expert-trace source with verified target-correct hard recoveries.
3. Train the next SFT checkpoint on the improved recovery traces.
4. Re-run reset-aware RLOO only after SFT produces a stronger correctness signal on hard prompts.
5. Run a broader raw-versus-reset-aware comparison on a larger hard Countdown slice.
6. Scale the extension comparison experiments after the one-shot baseline becomes more arithmetically reliable.

## Recommended Repo Entry Points

- `README.md`
  - best quick overview of what the repository currently supports
- `docs/code-walkthrough.md`
  - best file-by-file explanation of the code and terminology
- `docs/status-report.md`
  - best detailed engineering status snapshot
- `src/learning_to_reset/`
  - main implementation directory for the baseline and extension modules
- `tests/`
  - regression coverage for the current pipeline

## Short Teammate Takeaway

The project is no longer just a proposal plus notes. It now has a working baseline pipeline, real experiment artifacts, measurable reset-aware gains in output validity, and an implemented extension track. The next phase is mainly execution on better arithmetic-grounded supervision rather than missing infrastructure.

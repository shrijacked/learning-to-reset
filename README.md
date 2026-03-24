# Learning to Reset

This repository explores dynamic context management for mathematical reasoning with a `<clean>` token, a one-shot reset manager, and reward shaping for reset-aware training.

## Current Scope

The repository currently covers three core software primitives:

- SFT trace curation that teaches the model when to emit `<clean>`
- A one-shot context manager that handles `y0 -> optional <clean> -> y1`
- Modified RLOO utilities that propagate the final reward through both segments

This is an initial research scaffold, not a full training pipeline yet. The next major milestone is integrating expert traces, Countdown data, and model training/evaluation loops.

## Project Docs

- `docs/method-overview.md`: current method summary and implementation targets
- `docs/project-plan.md`: architecture, task traceability, and verification plan
- `docs/extension-roadmap.md`: future direction for multi-step cleaning and memory-aware control
- `skills/learning-to-reset-research/SKILL.md`: repo-local working guide for future development

## Quick Start

Run the tests:

```bash
python3 -m unittest discover -s tests -v
```

## Repository Layout

```text
docs/                      Paper summary, plan, and extension roadmap
skills/                    Repo-local skill to keep future work aligned to the project direction
src/learning_to_reset/     Core context-reset utilities
tests/                     Regression tests for the current behavior
.github/workflows/         GitHub CI
```

## Immediate Next Steps

1. Integrate the expert trace format used for SFT.
2. Add dataset adapters for Countdown and leaderboard-style evaluation.
3. Connect the context manager and modified RLOO math to a training loop.
4. Extend the one-shot cleaner toward multi-step cleaning and selective memory retention.

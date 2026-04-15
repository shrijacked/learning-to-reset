# Learning to Reset

This repository explores dynamic context management for mathematical reasoning with a `<clean>` token, a one-shot reset manager, and reward shaping for reset-aware training.

## Current Scope

The repository currently covers three core software primitives:

- SFT trace curation that teaches the model when to emit `<clean>`
- A one-shot context manager that handles `y0 -> optional <clean> -> y1`
- Modified RLOO utilities that propagate the final reward through both segments
- A synthetic Countdown-aligned fallback trace generator backed by a deterministic solver
- Multi-solution synthetic supervision and arithmetic walkthrough trace generation
- A bounded multi-clean extension with a reset budget and per-clean penalty
- Dataset loaders and prompt builders for trace and Countdown-style records
- Deterministic split and batching helpers for future training/evaluation loops
- JSONL artifact export and CLI preparation command for trainer-ready splits
- Countdown answer verification for evaluating generated expressions
- SFT runtime for prepared supervised traces
- Clean-aware reward and trajectory assembly for Countdown rollouts
- Reset-aware RLOO runtime with local metrics and checkpointing
- Clean-aware evaluation that retries once after `<clean>` by default

The repository now supports the baseline pipeline over prepared artifacts: fetch paper-aligned source files, prepare data, run SFT, run reset-aware RLOO, and evaluate with the one-shot clean retry path. The main remaining work is scaling those real-source paths into stronger paper-aligned experiments and target-model runs.

The latest local pilot now exercises that path on Countdown-aligned fallback traces and real fetched Countdown prompts. The current small CPU-only baseline still scores `0/4` on the held-out slice for target correctness, but it now shows a stable gap between raw one-pass decoding and reset-aware retries: raw generation remains invalid on the hard slice, while reset-aware evaluation reliably reaches valid post-clean arithmetic expressions with `clean_rate = 1.0`.

## Project Docs

- `docs/method-overview.md`: current method summary and implementation targets
- `docs/code-walkthrough.md`: file-by-file explanation of the code and jargon
- `docs/current-approaches.md`: summary of current methods and the project gap
- `docs/project-plan.md`: architecture, task traceability, and verification plan
- `docs/extension-roadmap.md`: future direction for multi-step cleaning and memory-aware control
- `docs/status-report.md`: completed work, missing engineering tasks, and non-engineering leftovers
- `docs/team-summary.md`: short shareable snapshot for collaborators
- `skills/learning-to-reset-research/SKILL.md`: repo-local working guide for future development

## Quick Start

Run the tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Run the baseline mechanics demo:

```bash
PYTHONPATH=src python3 -m learning_to_reset
```

Install the local training stack:

```bash
python3 -m venv .venv
./.venv/bin/pip install transformers datasets accelerate torch
```

Prepare split JSONL artifacts from raw trace and Countdown files:

```bash
PYTHONPATH=src python3 -m learning_to_reset.prepare_artifacts \
  --traces path/to/traces.jsonl \
  --countdown path/to/countdown.jsonl \
  --output-dir output/prepared
```

Fetch a paper-aligned source bundle and prepare real-source artifacts:

```bash
PYTHONPATH=src ./.venv/bin/python -m learning_to_reset.paper_sources \
  --output-dir tmp/paper-assets \
  --max-train-samples 12 \
  --max-eval-samples 4 \
  --max-reference-trace-rows 64

PYTHONPATH=src ./.venv/bin/python -m learning_to_reset.prepare_paper_artifacts \
  --traces tmp/paper-assets/reference-traces.jsonl \
  --countdown-train tmp/paper-assets/countdown-train.jsonl \
  --countdown-eval tmp/paper-assets/countdown-eval.jsonl \
  --output-dir tmp/paper-artifacts
```

Generate a Countdown-aligned fallback trace set directly from local Countdown prompts:

```bash
PYTHONPATH=src ./.venv/bin/python -m learning_to_reset.synthetic_countdown_traces \
  --countdown tmp/paper-assets/countdown-train.jsonl \
  --output-path tmp/paper-assets/synthetic-countdown-traces.jsonl \
  --solutions-per-sample 4
```

Run SFT on prepared artifacts:

```bash
PYTHONPATH=src ./.venv/bin/python -m learning_to_reset.sft_runtime \
  --train output/prepared/sft-train.jsonl \
  --validation output/prepared/sft-validation.jsonl \
  --model path/or/model-name \
  --output-dir output/checkpoints/sft
```

Run Countdown generation and scoring:

```bash
PYTHONPATH=src ./.venv/bin/python -m learning_to_reset.eval_runtime \
  --prepared-countdown output/prepared/countdown-test.jsonl \
  --model path/or/model-name \
  --output-dir output/eval/countdown
```

Run reset-aware RLOO on prepared Countdown prompts:

```bash
PYTHONPATH=src ./.venv/bin/python -m learning_to_reset.rloo_runtime \
  --train output/prepared/countdown-train.jsonl \
  --validation output/prepared/countdown-validation.jsonl \
  --model output/checkpoints/sft \
  --output-dir output/checkpoints/rloo \
  --responses-per-prompt 4
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

1. Replace the fallback trace source with a stronger paper-native Countdown expert-trace source.
2. Strengthen the recovery supervision further so post-clean retries move from valid-looking expressions to arithmetic-consistent, target-correct expressions.
3. Scale the pilot from tiny local subsets to a larger paper-style train/eval run.
4. Re-run the base, SFT, and reset-aware checkpoints on the same held-out hard Countdown slice, including raw vs reset-aware comparisons.
5. Extend the new bounded multi-clean path toward selective retention and recall-aware memory.

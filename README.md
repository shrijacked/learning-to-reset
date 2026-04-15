# Learning to Reset

This repository explores dynamic context management for mathematical reasoning with a `<clean>` token, a one-shot reset manager, and reward shaping for reset-aware training.

## Current Scope

The repository currently covers three core software primitives:

- SFT trace curation that teaches the model when to emit `<clean>`
- A one-shot context manager that handles `y0 -> optional <clean> -> y1`
- Modified RLOO utilities that propagate the final reward through both segments
- A synthetic Countdown-aligned fallback trace generator backed by a deterministic solver
- Multi-solution synthetic supervision and arithmetic walkthrough trace generation
- Verifier-grounded and contrastive synthetic recovery responses for arithmetic-consistency supervision
- A bounded multi-clean extension with a reset budget and per-clean penalty
- Dataset loaders and prompt builders for trace and Countdown-style records
- Deterministic split and batching helpers for future training/evaluation loops
- JSONL artifact export and CLI preparation command for trainer-ready splits
- Countdown answer verification for evaluating generated expressions
- SFT runtime for prepared supervised traces
- Clean-aware reward and trajectory assembly for Countdown rollouts
- Reset-aware RLOO runtime with local metrics and checkpointing
- Clean-aware evaluation that retries once after `<clean>` by default
- Per-example evaluation diagnostics that record the verifier-computed expression value
- A comparison utility for raw one-pass versus reset-aware evaluation summaries
- An opt-in verifier gate for retry-stage recovery examples during SFT artifact preparation

The repository now supports the baseline pipeline over prepared artifacts: fetch paper-aligned source files, prepare data, run SFT, run reset-aware RLOO, and evaluate with the one-shot clean retry path. The main remaining work is improving arithmetic grounding and scaling those real-source paths into stronger target-model runs.

The latest local pilot now exercises that path on real fetched Countdown prompts plus expanded Countdown-aligned fallback traces. The best current CPU-only checkpoint is the expanded SFT run: raw one-pass decoding remains `0/8` valid, while reset-aware evaluation reaches `8/8` valid, `1/8` correct, and `clean_rate = 1.0` on the held-out slice. A balanced verifier-grounded ablation ties that result, while a small reset-aware RLOO pass completed but did not improve held-out correctness.

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
  --output-dir tmp/paper-artifacts \
  --include-recovery-examples \
  --require-recovery-target-correct
```

Generate a Countdown-aligned fallback trace set directly from local Countdown prompts:

```bash
PYTHONPATH=src ./.venv/bin/python -m learning_to_reset.synthetic_countdown_traces \
  --countdown tmp/paper-assets/countdown-train.jsonl \
  --output-path tmp/paper-assets/synthetic-countdown-traces.jsonl \
  --solutions-per-sample 4 \
  --recovery-style all
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

Compare raw one-pass and reset-aware evaluation runs:

```bash
PYTHONPATH=src ./.venv/bin/python -m learning_to_reset.compare_eval_results \
  --baseline-dir output/eval/countdown-raw \
  --candidate-dir output/eval/countdown-reset-aware \
  --output-dir output/eval/comparison
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

1. Use the strict recovery verifier gate for the next prepared SFT artifact set.
2. Add or acquire a stronger Countdown-native expert-trace source if strict recovery filtering still does not improve correctness.
3. Re-run reset-aware RLOO only after SFT improves beyond the current `1/8` held-out correctness result.
4. Scale from local CPU pilots to a larger target-model train/eval run.
5. Extend the new bounded multi-clean path toward selective retention and recall-aware memory.

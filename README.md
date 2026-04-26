# Learning to Reset

This repository explores dynamic context management for mathematical reasoning with a `<clean>` token, a one-shot reset manager, and reward shaping for reset-aware training.

## Current Scope

The repository currently covers three core software primitives:

- SFT trace curation that teaches the model when to emit `<clean>`
- A one-shot context manager that handles `y0 -> optional <clean> -> y1`
- Modified RLOO utilities that propagate the final reward through both segments
- A synthetic Countdown-aligned fallback trace generator backed by a deterministic solver
- Multi-solution synthetic supervision and arithmetic walkthrough trace generation
- Verifier-grounded, contrastive, and arithmetic-grounded synthetic recovery responses (including the `grounded` style with substep arithmetic, rejected hypothesis, number-budget audit, and final-value reconciliation) for arithmetic-consistency supervision
- A deep-verify filter that re-checks every inline `a op b = c` claim in mined or synthetic recoveries before they enter SFT
- Hard-focused synthetic Countdown sample generation for multiplication/division-heavy training data
- Failure-mined recovery trace generation from failed hard eval outputs
- A bounded multi-clean extension with a reset budget and per-clean penalty
- A selective-retention extension that carries explicit `<retain>...</retain>` notes across clean resets
- A recall-aware memory extension with explicit `<memory>...</memory>` writes, deterministic recall, and a memory-aware clean loop
- A comparison utility for full reset, selective retention, and memory-aware clean trajectories
- Dataset loaders and prompt builders for trace and Countdown-style records
- Deterministic split and batching helpers for future training/evaluation loops
- JSONL artifact export and CLI preparation command for trainer-ready splits
- Countdown answer verification for evaluating generated expressions
- A deterministic hard Countdown eval slice for multiplication/division-heavy comparisons
- SFT runtime for prepared supervised traces
- Clean-aware reward and trajectory assembly for Countdown rollouts
- Reset-aware RLOO runtime with local metrics and checkpointing
- Clean-aware evaluation that retries once after `<clean>` by default
- Per-example evaluation diagnostics that record the verifier-computed expression value
- A comparison utility for raw one-pass versus reset-aware evaluation summaries
- An opt-in verifier gate for retry-stage recovery examples during SFT artifact preparation
- A multi-clean evaluation decoder (`--max-clean-tries N`) with verifier-aware first-correct early stop
- `score_when_cleaned` and `clean_rate` metrics in the eval summary, aligned with paper Figure 6
- A contamination guard on `failure_recovery_traces` that aborts mining when source IDs overlap a held-out slice
- A Figure 7-style qualitative best/worst sample exporter (`scripts/export_qualitative_samples.py`)
- A `--scale {pilot,paper}` preset on `paper_sources` and `prepare_paper_artifacts`
- An end-to-end `scripts/replicate_paper.sh` pipeline with `--dry-run` smoke mode

The repository now supports the baseline pipeline over prepared artifacts: fetch paper-aligned source files, prepare data, run SFT, run reset-aware RLOO, and evaluate with the one-shot clean retry path. The main remaining work is improving arithmetic grounding and scaling those real-source paths into stronger target-model runs.

The latest local pilot now exercises that path on real fetched Countdown prompts plus expanded Countdown-aligned fallback traces. The best current CPU-only checkpoint is the expanded SFT run: raw one-pass decoding remains `0/8` valid, while reset-aware evaluation reaches `8/8` valid, `1/8` correct, and `clean_rate = 1.0` on the held-out slice. A balanced verifier-grounded ablation ties that result. Hard-focused SFT and a hard-focused reset-aware RLOO pass both completed, but neither improved target-correct arithmetic on the original 3-example hard slice. A plus-mined SFT run on a fresh 32-example hard holdout scored `29/32` valid and `1/32` correct with reset-aware retry.

## Project Docs

- `docs/method-overview.md`: current method summary and implementation targets
- `docs/code-walkthrough.md`: file-by-file explanation of the code and jargon
- `docs/current-approaches.md`: summary of current methods and the project gap
- `docs/project-plan.md`: architecture, task traceability, and verification plan
- `docs/extension-roadmap.md`: future direction for multi-step cleaning and memory-aware control
- `docs/status-report.md`: completed work, missing engineering tasks, and non-engineering leftovers
- `docs/team-summary.md`: short shareable snapshot for collaborators
- `docs/paper-claims-traceability.md`: every `main.pdf` Section 3/4 claim mapped to the file and test that implements it
- `docs/paper-replication.md`: hardware, wall-clock, expected hard-Countdown band, and verification checklist for the full paper-faithful run
- `docs/grounded-recovery-results-2026-04-26.md`: arithmetic-grounded recovery experiment write-up
- `docs/diagrams/end-to-end-pipeline.html`: post-Phase-A architecture diagram
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
./.venv/bin/pip install -e .[trainer]
```

The `trainer` extra installs `transformers`, `datasets`, `accelerate`, and `torch`. Skip it if you only want to run the unit tests, which work on the standard library alone.

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

This writes both `countdown-test.jsonl` and `countdown-test-hard.jsonl`; the hard slice keeps examples that require multiplication or division under the deterministic Countdown solver.

Generate a Countdown-aligned fallback trace set directly from local Countdown prompts:

```bash
PYTHONPATH=src ./.venv/bin/python -m learning_to_reset.synthetic_countdown_dataset \
  --reference-path tmp/paper-assets/countdown-train.jsonl \
  --output-path tmp/paper-assets/countdown-train-hard.jsonl \
  --num-samples 64 \
  --require-hard

PYTHONPATH=src ./.venv/bin/python -m learning_to_reset.synthetic_countdown_traces \
  --countdown tmp/paper-assets/countdown-train-hard.jsonl \
  --output-path tmp/paper-assets/synthetic-countdown-traces.jsonl \
  --solutions-per-sample 4 \
  --recovery-style all
```

`--recovery-style all` now includes the `grounded` style alongside `walkthrough`, `verification`, and `contrastive`. Use `--recovery-style grounded` to emit only the arithmetic-grounded template with substep arithmetic, a rejected hypothesis, a number-budget audit, and a final-value reconciliation.

Mine failed hard eval outputs into solver-verified recovery traces for the next training cycle:

```bash
PYTHONPATH=src ./.venv/bin/python -m learning_to_reset.failure_recovery_traces \
  --prepared-countdown output/prepared/countdown-test-hard.jsonl \
  --eval-results output/eval/countdown-hard/results.jsonl \
  --output-path output/traces/mined-hard-recoveries.jsonl \
  --max-solutions-per-failure 3 \
  --recovery-style all
```

Use mined traces as training data only with a fresh held-out comparison slice; do not report metrics on the same examples that were mined into recovery supervision.

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

Compare extension trajectories in code with `compare_extension_trajectories(...)`, then write JSON/Markdown artifacts with `write_extension_comparison_outputs(...)`. From the CLI, score the `full_reset`, `selective_retention`, and `memory` controllers on the segments that an existing multi-clean eval already produced:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/run_extension_comparison.py \
  --prepared-countdown output/prepared/countdown-test-hard.jsonl \
  --results-jsonl output/eval/multi-clean-3/results.jsonl \
  --output-dir output/eval/extensions \
  --max-cleans 3
```

Run the verifier-aware multi-clean evaluator end-to-end (Figure 6 metrics):

```bash
PYTHONPATH=src ./.venv/bin/python -m learning_to_reset.eval_runtime \
  --prepared-countdown output/prepared/countdown-test-hard.jsonl \
  --model output/checkpoints/sft \
  --output-dir output/eval/multi-clean-3 \
  --max-new-tokens 384 \
  --max-clean-tries 3
```

If accuracy is stuck at zero but `average_score` or `score_when_cleaned` hovers around **0.1**, the model is usually emitting **legal** `<answer>` expressions that simply **miss the target** (scoring gives 0.1 partial credit for validity + 1.0 only when correct). For retries, add **`--verifier-feedback`** so each post-`<clean>` prompt includes the verifier’s computed value vs target; that is stronger than re-sending the same bare question and is the first decode-time lever to try before larger models or RLOO shaped on verifier reward.

Export Figure 7-style best/worst qualitative samples from any eval directory:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/export_qualitative_samples.py \
  --eval-results output/eval/multi-clean-3/results.jsonl \
  --output-path output/eval/multi-clean-3/qualitative.md
```

Drive the full paper-faithful pipeline (requires a 1B+ GPU; see `docs/paper-replication.md`):

```bash
PYTHON=python3 PYTHONPATH=src bash scripts/replicate_paper.sh \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --out-dir runs/replicate-paper-2026-04-26
```

Smoke-validate the same pipeline locally on Qwen 0.5B without any model run:

```bash
PYTHON=python3 PYTHONPATH=src bash scripts/replicate_paper.sh --dry-run \
  --base-model Qwen/Qwen2.5-0.5B \
  --out-dir /tmp/replicate-paper-dryrun
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

1. Improve arithmetic verification in the recovery traces so the model stops writing plausible but false “verified” equations.
2. Add or fetch a stronger Countdown-native expert-trace source with verified target-correct recoveries.
3. Scale from local CPU pilots to a larger target-model train/eval run once stronger traces are available.
4. Run a broader raw-versus-reset-aware comparison on a larger hard Countdown slice.
5. Use the extension comparison utility to scale full reset, selective retention, and memory-aware clean-loop comparisons.

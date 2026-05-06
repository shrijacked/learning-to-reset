# Countdown-First SFT Patch Proposal

## Purpose

This document specifies a patch to prevent task dilution in the training pipeline by making supervised fine-tuning (SFT) primarily Countdown-native while preserving reset/recovery behavior learning.

The patch is designed for this repository and maps directly to existing modules:

- `scripts/replicate_paper.sh`
- `src/learning_to_reset/paper_sources.py`
- `src/learning_to_reset/prepare_paper_artifacts.py`
- `src/learning_to_reset/paper_dataset_prep.py`
- `src/learning_to_reset/synthetic_countdown_traces.py`
- `src/learning_to_reset/merge_sft_corpus.py`
- `src/learning_to_reset/sft_runtime.py`

---

## Problem Statement

The current run path can underperform on Countdown because SFT data is dominated by non-Countdown traces and can include malformed combined rows during step-6 merge.

### Observed symptoms from local run artifacts

From `outputs/replicate-paper`:

- `artifacts/sft-train.jsonl` has 115 rows sourced from reference traces.
- `artifacts/countdown-train.jsonl` has 230 rows but these are prompt-only eval/RL artifacts, not SFT traces.
- `sft-train-combined.jsonl` has 138 rows, with base rows dominated by non-Countdown source ids and appended mined rows.
- `eval-raw/summary.json` reports `accuracy = 0.0`, `valid_rate = 0.0` on the sampled subset.

### Why this is a dilution issue

SFT is where behavior format and local reasoning style are learned. If that phase is mostly off-domain, the model can learn good structural behavior (tags, generic self-correction phrasing) without learning Countdown arithmetic patterns with sufficient density.

In a small-model setting, domain mismatch in SFT can dominate early optimization and hurt downstream RL bootstrap quality.

---

## Paper Alignment

`researchpaper.md` supports Countdown-aligned priming:

- Priming/SFT trajectories are generated on Countdown tasks.
- Behavioral control (verification/backtracking/subgoal/backward-chaining) is applied within that task context.
- Curated OpenWebMath interventions are a separate pretraining intervention, not a replacement for Countdown priming.

Therefore, a Countdown-first SFT patch is consistent with the methodology this project is trying to replicate.

---

## Patch Goals

1. Make SFT training examples Countdown-first by default.
2. Keep recovery/reset behavior supervision.
3. Preserve optional off-domain behavior traces as a small regularizer.
4. Enforce strict corpus schema so malformed rows cannot silently enter SFT.
5. Keep the pipeline reproducible and controllable by flags.

---

## Non-Goals

- Replacing RL objective logic in `rloo_runtime.py`.
- Changing evaluation metrics or verifier semantics.
- Rewriting the entire paper replication pipeline.
- Introducing external dependencies.

---

## High-Level Design

Introduce a new SFT source mode in the replication pipeline:

- `reference` (current behavior; baseline compatibility)
- `synthetic-countdown` (new default for pilot runs)
- `mixed` (countdown synthetic + sampled reference traces)

The core idea:

1. Generate Countdown-native trace records from Countdown train source file using `synthetic_countdown_traces`.
2. Feed those traces into `prepare_paper_artifacts --traces`.
3. Merge mined recoveries through `merge_sft_corpus` only after strict conversion to PromptExample rows.
4. Validate all SFT JSONL files against prompt/response schema before any SFT call.

---

## Detailed Changes

## 1) `scripts/replicate_paper.sh`

### New flags

- `--sft-source-mode {reference,synthetic-countdown,mixed}`
- `--sft-mix-reference-ratio <float in [0,1]>` (only used in `mixed`)
- `--synthetic-max-samples <int|unset>`
- `--synthetic-solutions-per-sample <int>`
- `--synthetic-recovery-style {walkthrough,verification,contrastive,grounded,both,all}`

### New default behavior

- Pilot/default path should use `synthetic-countdown`.
- Paper-faithful path may keep `reference` if strict reproduction is desired.
- Explicit mode selection must always override defaults.

### Step updates

- Step 1 remains source fetch.
- Insert new Step 1.5:
  - Generate synthetic Countdown traces from `sources/countdown-train.jsonl`.
  - Output `sources/sft-synthetic-traces.jsonl`.
- For `mixed` mode:
  - Build mixed traces file with deterministic sampling from reference and synthetic.
- Step 2 changes:
  - Pass selected traces file to `prepare_paper_artifacts --traces`.
- Before Step 3 and Step 6 SFT:
  - Run schema validator on train and validation JSONL.

---

## 2) `src/learning_to_reset/prepare_paper_artifacts.py`

### Add explicit metadata to manifest

Include:

- `trace_source_mode`
- `trace_source_path`
- `trace_source_counts` (source-domain counts if available)

### Add pass-through controls

If needed for downstream CLI compatibility:

- `--trace-source-mode` as metadata annotation only.

---

## 3) `src/learning_to_reset/paper_dataset_prep.py`

### Enforce trace domain tags in output metadata

When converting traces to PromptExamples:

- Add `metadata["trace_domain"]`:
  - `countdown-synthetic`
  - `reference-behavior`
  - `mined-recovery`

This enables post-hoc composition analysis from one JSONL file.

### Maintain deterministic split behavior

Keep existing deterministic split logic, but write optional source-domain stats to `manifest.json`.

---

## 4) `src/learning_to_reset/merge_sft_corpus.py`

### Hard requirement

Merged output must contain only PromptExample-shaped rows:

- required key: `prompt` (non-empty string)
- optional key: `response` (string; may be empty)
- optional key: `metadata` (object)

### Add row validation

Before writing:

- validate base examples
- validate mined-converted examples
- fail with clear message on first invalid row index

### Add domain annotation for mined examples

Set:

- `metadata["trace_domain"] = "mined-recovery"`
- `metadata["stage"] = "retry-recovery"` for recovery examples

---

## 5) `src/learning_to_reset/sft_runtime.py`

### Add strict loader mode

Enhance `load_prepared_examples`:

- Validate `prompt` presence/type and non-empty content.
- Validate `response` type when present.
- Validate `metadata` object type.
- Raise error with row number and path on violation.

Add CLI flag:

- `--strict-input-schema` (default on in pipeline script).

This prevents silent acceptance of malformed JSONL rows.

---

## 6) `src/learning_to_reset/synthetic_countdown_traces.py`

### Reuse as primary SFT source

No architectural rewrite required. Use existing generator with configuration tuned for SFT priming:

- `include_negative=True`
- `recovery_style=grounded` (default for this patch)
- `solutions_per_sample=1` (start conservative; ablate later)

### Optional enhancement

Add a mode that emits behavior-balanced templates (verification/backtracking distribution targets) if needed after initial patch.

---

## Data Flow After Patch

1. Fetch `countdown-train`, `countdown-eval`, and reference traces.
2. Generate `sft-synthetic-traces.jsonl` from `countdown-train`.
3. Select trace source file by mode:
   - `reference`: existing reference traces
   - `synthetic-countdown`: synthetic traces
   - `mixed`: deterministic mixture file
4. Prepare artifacts:
   - `sft-train.jsonl`, `sft-validation.jsonl`
   - Countdown train/val/test files unchanged
5. First SFT on validated `sft-train.jsonl`.
6. Mine failures from Countdown eval.
7. Merge mined traces into PromptExamples with strict validation.
8. Second SFT on validated `sft-train-combined.jsonl`.
9. RLOO and final eval unchanged.

---

## Configuration Matrix

## Recommended defaults

### Pilot (CPU / small model)

- `sft-source-mode=synthetic-countdown`
- `synthetic-max-samples=256` (or current cap equivalent)
- `synthetic-solutions-per-sample=1`
- `synthetic-recovery-style=grounded`
- `sft-mix-reference-ratio=0.0`

### Paper-faithful compatibility

- `sft-source-mode=reference`
- existing scale knobs unchanged

### Exploratory mixed mode

- `sft-source-mode=mixed`
- `sft-mix-reference-ratio=0.1` (start low)

---

## Validation Plan

## A) Schema and composition checks (must pass)

Add scriptable checks after step-2 and step-6:

- Every SFT row has non-empty `prompt`.
- No non-PromptExample keys at top level in SFT files.
- Composition report:
  - total rows
  - rows by `trace_domain`
  - rows containing `<clean>`
  - rows containing `<answer>`

## B) Functional checks

Run existing dry-run tests plus:

- smoke run with `synthetic-countdown` mode
- smoke run with `mixed` mode
- smoke run with `reference` mode for backward compatibility

## C) Efficacy checks

For each mode (`reference`, `synthetic-countdown`, `mixed-0.1`) run:

1. Step-4 raw eval accuracy/validity on same capped set.
2. Step-8 final hard-slice accuracy and clean-rate.
3. Report training stability:
   - NaN/inf loss incidence
   - eval loss trend

Use same random seed and same model for comparison.

---

## Acceptance Criteria

Patch is accepted only if all are true:

1. Strict schema validation blocks malformed SFT corpora.
2. `synthetic-countdown` mode produces SFT data where Countdown-domain rows are dominant.
3. `reference` mode remains runnable (backward compatibility).
4. On pilot-cap experiments, `synthetic-countdown` is not worse than `reference` on:
   - step-4 valid rate
   - step-8 hard-slice accuracy
5. Documentation and CLI help text reflect the new controls.

---

## Risk Analysis

## Risk 1: Overfitting to synthetic template style

Mitigations:

- keep `solutions_per_sample` and style configurable
- add optional style randomization
- test `mixed` mode with low off-domain ratio

## Risk 2: Losing beneficial generic reasoning priors

Mitigations:

- keep `mixed` mode
- include reference-ratio ablation (0.05, 0.1, 0.2)

## Risk 3: Increased pipeline complexity

Mitigations:

- mode enum is small and explicit
- defaults remain simple
- manifest records selected mode and counts

## Risk 4: Reproducibility drift

Mitigations:

- deterministic sampling/shuffling for mixed mode
- write source counts and seed into manifest

---

## Rollout Plan

## Phase 1: Safety rails first

- Implement strict schema validation in `sft_runtime` and `merge_sft_corpus`.
- Add composition report utility.

## Phase 2: New trace source modes

- Add mode flags and synthetic generation step in script.
- Wire selected traces into `prepare_paper_artifacts`.

## Phase 3: Controlled experiments

- Run 3-mode ablation on pilot caps.
- Choose repository default based on hard-slice metric and stability.

---

## Rollback Plan

Rollback is immediate via CLI flag:

- set `--sft-source-mode reference`
- disable strict schema only if absolutely necessary (not recommended)

No data migration is required; old artifacts remain valid for old mode.

---

## Implementation Checklist

- [ ] Add `sft-source-mode` and synthetic controls to `scripts/replicate_paper.sh`
- [ ] Generate synthetic trace file when mode requires it
- [ ] Add mixed-mode deterministic sampler
- [ ] Wire selected trace file to `prepare_paper_artifacts --traces`
- [ ] Add strict schema checks in `merge_sft_corpus.py`
- [ ] Add strict schema checks in `sft_runtime.py`
- [ ] Add trace-domain metadata in dataset prep outputs
- [ ] Extend `manifest.json` with mode and composition fields
- [ ] Add/extend tests for:
  - [ ] mode parsing and defaults
  - [ ] schema rejection behavior
  - [ ] mixed-mode deterministic ratio behavior
  - [ ] backward compatibility of `reference` mode
- [ ] Update `README.md` and `docs/paper-replication.md` examples

---

## Proposed Experiment Table Template

Use this table for patch evaluation:

| Mode | Ref Ratio | Step-4 Accuracy | Step-4 Valid Rate | Step-8 Hard Accuracy | Step-8 Clean Rate | Notes |
|------|-----------|-----------------|-------------------|----------------------|-------------------|-------|
| reference | 1.0 |  |  |  |  | baseline |
| synthetic-countdown | 0.0 |  |  |  |  | primary candidate |
| mixed | 0.1 |  |  |  |  | regularized |
| mixed | 0.2 |  |  |  |  | stress test |

---

## Final Recommendation

Adopt `synthetic-countdown` as the pilot/default SFT source mode, retain `reference` for strict legacy reproduction, and enforce strict SFT schema validation before every SFT stage.

This directly addresses the observed dilution path while remaining consistent with the research paper’s Countdown-priming methodology.


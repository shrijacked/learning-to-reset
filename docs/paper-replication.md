# Paper Replication Runbook

This document is the operating manual for running the full Section 4
pipeline from `main.pdf` ("Learning to Reset"). It is written so a
collaborator with GPU hardware can reproduce the paper's hard-Countdown
correctness jump (13.61% → 36.94%) end to end with a single command.

The runbook pairs with `scripts/replicate_paper.sh`, which orchestrates
the ten pipeline steps, and `docs/paper-claims-traceability.md`, which
maps each paper claim to its implementation.

> **Why this exists.** The "Learning to Reset" project's local pilot
> uses Qwen-2.5 0.5B on CPU/MPS, which the paper notes plateaus around
> 1/32 hard correctness without paper-scale data and epochs. The
> replication path documented here is for users who have GPU access
> and want to reproduce the published numbers.

## 1. Hardware and software floor

| Resource             | Minimum                              | Recommended           |
|----------------------|--------------------------------------|-----------------------|
| GPU                  | 1× A100 40 GB                        | 1× A100 80 GB         |
| Host RAM             | 64 GB                                | 128 GB                |
| Disk                 | 100 GB free                          | 250 GB free           |
| CUDA                 | 12.x                                 | 12.4 or newer         |
| PyTorch              | ≥ 2.3                                | match Hugging Face    |
| Transformers         | ≥ 4.42                               | latest stable         |
| Python               | 3.10–3.12                            | 3.11                  |

CPU/MPS is **not supported** for paper-faithful replication. It works
for the dry-run smoke test (Section 6) but not for the SFT or RLOO
training stages at paper scale.

Install the project with the trainer extras:

```bash
git clone <repo-url> learning-to-reset
cd learning-to-reset
python -m venv .venv && source .venv/bin/activate
pip install -e .[trainer]
```

## 2. Wall-clock estimate

| Stage                                       | A100 40GB | A100 80GB |
|---------------------------------------------|-----------|-----------|
| 1. Fetch sources (HF download)              | ~5 min    | ~5 min    |
| 2. Prepare artifacts (`--scale paper`)      | ~10 min   | ~10 min   |
| 3. SFT pass 1 (1 epoch, 60K traces)         | ~6 h      | ~3 h      |
| 4. Baseline raw eval (3K examples)          | ~30 min   | ~20 min   |
| 5. Mine failures + deep-verify              | ~5 min    | ~5 min    |
| 6. SFT pass 2 (combined corpus)             | ~6 h      | ~3 h      |
| 7. Reset-aware RLOO (200 steps × 4 rollouts)| ~10 h     | ~5 h      |
| 8. Final multi-clean eval                   | ~45 min   | ~30 min   |
| 9. Extension comparison                     | ~30 min   | ~20 min   |
| 10. Qualitative export                      | < 1 min   | < 1 min   |
| **Total**                                   | **~24 h** | **~12 h** |

Numbers are based on the paper's reported recipe (Section 4.1) plus
local timing scaled by GPU FLOP estimates. Yours may vary ±30%.

## 3. Expected outcome

The headline target is the hard-Countdown correctness band reported in
Section 4.3 of `main.pdf`:

| Slice                       | Baseline (raw)  | Reset-aware (paper) |
|-----------------------------|-----------------|---------------------|
| `countdown-test-hard.jsonl` | **~13.61 %**    | **~36.94 %**        |
| `countdown-test.jsonl`      | ~25 % (paper)   | ~50 % (paper)       |

Acceptance threshold for "replicates the paper": final reset-aware
hard-correctness ≥ 30% on `countdown-test-hard.jsonl` after Step 8.
Below 25% indicates a misconfigured run; investigate before reporting.

The pipeline also reports `clean_rate`, `score_when_cleaned`, and the
extension comparison numbers used to plot Figures 6 and 7. The
qualitative export (Step 10) writes ten best/ten worst cases to
`qualitative.md` for direct comparison against Figure 7.

## 4. One-command run

```bash
bash scripts/replicate_paper.sh \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --out-dir runs/replicate-paper-$(date +%Y-%m-%d)
```

The script writes to a single `--out-dir`. Layout:

```
runs/replicate-paper-YYYY-MM-DD/
├── sources/                   # raw HF datasets, paper scale
├── artifacts/                 # prepared SFT + Countdown JSONL
├── sft/                       # checkpoint after pass 1
├── mined-recovery-traces.jsonl# deep-verify-filtered failures
├── sft-train-combined.jsonl   # sft-train + mined recoveries
├── sft-mined/                 # checkpoint after pass 2
├── rloo/                      # final reset-aware policy
├── eval-raw/                  # raw baseline eval (Step 4)
├── eval-final/                # multi-clean reset-aware eval (Step 8)
├── extensions/                # Step 9 outputs (when present)
└── qualitative.md             # Figure-7 aligned best/worst export
```

## 5. Step-by-step verification checklist

After each step, confirm the listed outputs exist and have the right
shape. Failures here mean re-run that step before moving on.

| Step | Path                                         | Quick check                                      |
|------|----------------------------------------------|--------------------------------------------------|
| 1    | `sources/countdown-train.jsonl`              | `wc -l` ≈ 60_000                                  |
| 1    | `sources/countdown-eval.jsonl`               | `wc -l` ≈ 3_000                                   |
| 2    | `artifacts/sft-train.jsonl`                  | `wc -l` ≥ 50_000                                  |
| 2    | `artifacts/manifest.json`                    | `"scale": "paper"`                                |
| 3    | `sft/pytorch_model.bin` (or shards)          | `ls -la` ≥ 1.5 GB                                 |
| 4    | `eval-raw/summary.json`                      | `accuracy` ≈ 0.13–0.15                            |
| 5    | `mined-recovery-traces.jsonl`                | `wc -l` ≥ 1_000                                   |
| 6    | `sft-mined/pytorch_model.bin` (or shards)    | newer mtime than `sft/`                           |
| 7    | `rloo/eval_log.json`                         | last entry ≥ Step 3 acc + 0.05                    |
| 8    | `eval-final/summary.json`                    | `accuracy` ≥ 0.30 on hard split                   |
| 8    | `eval-final/summary.json`                    | `clean_rate` ≥ 0.5                                |
| 9    | `extensions/comparison.json`                 | three entries: full-reset / retention / memory    |
| 10   | `qualitative.md`                             | 10 best + 10 worst examples                       |

## 6. Dry-run smoke test (no GPU required)

The dry-run mode is what the local CPU/MPS pilot uses to verify that
every CLI parses before kicking off a paid run:

```bash
LTR_REPLICATE_DRY_RUN=1 bash scripts/replicate_paper.sh \
    --dry-run \
    --base-model Qwen/Qwen2.5-0.5B \
    --out-dir /tmp/replicate-paper-dryrun
```

Expected output: a series of `[dry-run] cli ok: ...` lines and a final
`dry-run complete.` message. Any `ERROR:` line means a CLI changed and
the script needs to be updated.

The repo's automated test suite includes a gated subprocess test
(`tests/test_replicate_paper_dry_run.py`) that runs this exact command
when `LTR_REPLICATE_DRY_RUN=1` is set, ensuring the script stays
aligned with the underlying CLIs after every commit.

## 7. Common failure modes

| Symptom                                              | Likely cause                                | Fix                                                  |
|------------------------------------------------------|---------------------------------------------|------------------------------------------------------|
| OOM during SFT pass 1                                | batch size too large for 40 GB              | `--gradient-accumulation-steps 4`                    |
| Tokenizer "regex pattern" warning                    | Mistral-style fix not relevant for Qwen     | safe to ignore                                       |
| `failure_recovery_traces` aborts on overlap          | source_ids leaked into train slice          | regenerate the holdout (`paper_sources --scale ...`) |
| `eval-final` accuracy plateaus at ~13%               | RLOO never converged                         | inspect `rloo/eval_log.json`; raise `--steps`        |
| Multi-clean decoder hangs                            | `--max-clean-tries` too high with greedy    | drop to `--max-clean-tries 2`                        |
| Score is high but `clean_rate` is 0                  | model never emits `<clean>`                 | recheck SFT data, ensure recovery examples included  |

## 8. What to report

When publishing replication results please include:

- The exact command used (including `--scale`, `--rloo-steps`, etc.).
- Hardware (GPU, CUDA version, PyTorch version).
- Final `eval-final/summary.json` (paste the JSON inline).
- Step-7 `rloo/eval_log.json` last 5 entries.
- The `qualitative.md` file (best/worst sampled cases).
- Any deviation from the recipe and why.

This mirrors the reproducibility checklist used in the original paper
and lets reviewers cross-check Figures 6 and 7 against your run.

## 9. Out of scope

- **CPU/MPS paper-faithful runs.** The local pilot at Qwen-2.5 0.5B
  reaches a hard-correctness ceiling around 1/32 even with all the
  decode-time levers; this is documented in
  `docs/grounded-recovery-results-2026-04-26.md`.
- **Live training of the 1B+ model from this project's CI.** The
  pipeline assumes you have provisioned GPU compute yourself.
- **Paper-faithful results without `--scale paper`.** The pilot scale
  preset caps every dataset and is intentionally lossy.

## 10. Contacts and provenance

- Code reference: `scripts/replicate_paper.sh`
- Claim-to-test mapping: `docs/paper-claims-traceability.md`
- Pipeline diagram: `docs/diagrams/end-to-end-pipeline.html`
- Audit notes: `docs/repo-audit-2026-04-25.md`
- Status: `docs/status-report.md`

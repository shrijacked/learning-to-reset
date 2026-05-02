# Learning to Reset — End-to-End Completion Plan

**Goal:** Turn this from "code that runs but doesn't visibly demonstrate the paper" into a polished, self-contained, working replication of the paper — with honest local results AND a real path to paper-faithful 1B numbers.

**Reality check we have to live with:** The paper's `36.94%` requires a 1B+ model on a real GPU. No amount of code work makes a 0.5B-on-CPU run hit that number. So the plan has two tracks: a "polish + working pilot" track that costs $0 and can start immediately, and a "real paper number" track that requires either ~$50-100 of cloud GPU time or a free-tier alternative.

---

## Phase 1 — Code health (1–2 days, $0, no decisions needed)

These close the five remaining audit findings and remove fragility from the pipeline.

| # | Action | File(s) | Why | Status |
|---|---|---|---|---|
| 1.1 | Rename `traces.jsonl` ↔ `reference-traces.jsonl` so they match | `replicate_paper.sh:202` | Cold-cache run currently breaks | ✅ Done |
| 1.2 | Validate `LTR_VERIFIER_FEEDBACK` env var; warn on common misspellings | `replicate_paper.sh` | Silent-disable trap | ✅ Done |
| 1.3 | Surface `skipped_missing_example` as a non-zero warning | `failure_recovery_traces.py` | Hides upstream split mismatches | ✅ Done |
| 1.4 | Fix trace curator silent-clean for incorrect-no-answer traces | `trace_curation.py:74-108` | Asymmetric data quality | ✅ Done |
| 1.5 | Add CPU-tiny integration test (full SFT→eval→mine→re-SFT loop) | new `tests/test_integration_pipeline.py` | First real end-to-end check | ✅ Done |
| 1.6 | Add provenance breakdown (per-source counts) to `manifest.json` | `paper_dataset_prep.py` | No audit trail on SFT mix | ✅ Done |
| 1.7 | Run extension comparison on seed=131 and seed=219 slices | `scripts/run_extension_comparison.py`, `docs/extension-comparison-seed-results.md` | Current claim based on one slice | ✅ Done |
| 1.8 | Verify 172+ tests still pass, push as one PR `code-health-cleanup` | — | Gate before Phase 2 | ⬜ Todo |

**Deliverable:** PR `code-health-cleanup`, all CI green, audit issues closed.

---

## Phase 2 — Demo + visualization (2–3 days, $0, no decisions needed)

This is the part that makes a visitor say "oh, I get it." The mechanism's success at 0.5B is *validity recovery*, and that needs to be visible.

### 2.1 Interactive demo notebook
New `notebooks/demo.ipynb`:
1. Load latest SFT checkpoint (from `tmp/paper-eval/...`)
2. Pick three hard Countdown problems
3. Generate **raw** response → show garbled / no `<answer>` output
4. Generate **reset-aware** response → show `<think>...<clean>` on first attempt, valid `<answer>` on retry
5. Run on 32-example holdout, print summary table (validity, correctness, clean rate)
6. Show one qualitative case where the model fabricates arithmetic (from `docs/grounded-recovery-results-2026-04-26.md`)

### 2.2 Figure pack (`docs/figures/`)
Generated from existing `tmp/paper-eval/` artifacts via new `scripts/build_figures.py`:
- **Fig A**: Bar chart — validity rate raw vs reset-aware across SFT variants
- **Fig B**: Hard correctness gap — 0/32 raw vs 1/32 reset-aware vs paper's 36.94%
- **Fig C**: Clean-rate distribution + score-when-cleaned per SFT variant
- **Fig D**: Token budget sweep (256 / 384 / 512) vs validity / correctness
- **Fig E**: Pipeline diagram rendered from `docs/diagram.md`

### 2.3 README rewrite
Replace the current README with a publication-quality front door:
1. One-paragraph what-and-why at the very top
2. Hero figure (pipeline + validity bar chart)
3. Results table:

   | Setup | Validity | Hard correct | Notes |
   |---|---|---|---|
   | raw 0.5B | 0/32 | 0/32 | no clean |
   | reset-aware 0.5B | 29/32 | 1/32 | local pilot ceiling |
   | paper 1B (their number) | — | 36.94% | Section 4.3 |
   | our 1B replication | — | ??? | see Phase 3 / runbook |

4. Try it yourself — 3 commands to run the demo notebook
5. Reproduction guide — link to `docs/paper-replication.md`
6. Honest results & limitations — links to `docs/grounded-recovery-results-2026-04-26.md`
7. Project structure — tree of `src/` with one-liner per module
8. Citations

### 2.4 Repo polish
- `Makefile` with `make demo`, `make test`, `make replicate-pilot`, `make replicate-paper`
- CI badge in README
- `LICENSE` check
- `CONTRIBUTING.md`

**Phase 2 implementation status:** notebook, figure builder, checked-in SVG figures,
README rewrite, Makefile, CI badge, MIT license, and contributing guide are in tree.

**Deliverable:** PRs `demo-notebook`, `figure-pack`, `readme-rewrite`. After merge, repo front page is shareable publicly.

---

## Phase 3 — GPU replication (1 day, team decides provider)

Only path to a real "we reproduced the paper" headline number. Team picks provider and budget.

| Provider | GPU | Cost | Notes |
|---|---|---|---|
| **RunPod** | A100 80GB | ~$1.89/hr | Cheapest A100. Pay-by-second, SSH access. |
| **Modal** | A100 80GB | ~$4.50/hr | Pay only while running; Python-native DX. |
| **Lambda Labs** | A100 40GB | ~$1.10/hr | Cheapest per-hour; often capacity-limited. |
| **Google Colab Pro+** | A100 (shared) | $50/mo | Only if subscription already paid. |
| **vast.ai** | A100 40GB | $0.80–1.50/hr | Cheapest variable-pricing; mixed reliability. |

Full pipeline at `--scale paper` ≈ 12–24h wall clock. Budget 1–2 runs = $25–100 total.

### 3.1 Pre-flight (free)
```bash
# Verify CLI parses at paper scale
LTR_REPLICATE_DRY_RUN=1 bash scripts/replicate_paper.sh \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --scale-override paper \
  --out-dir /tmp/dry-run
```

### 3.2 RunPod recipe (example)
```bash
# On your local machine
runpodctl create pod --gpuType "NVIDIA A100 80GB" --imageName "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"

# SSH into pod
git clone https://github.com/shrijacked/learning-to-reset && cd learning-to-reset
pip install -e ".[trainer]"
huggingface-cli login   # paste HF token

nohup bash scripts/replicate_paper.sh \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --scale-override paper \
  --out-dir /workspace/run \
  > /workspace/run.log 2>&1 &

tail -f /workspace/run.log
```

### 3.3 Post-run
- Update `docs/paper-replication.md` with actual numbers
- Update README results table with our 1B number
- If result is in `[20%, 40%]`: claim "replicated within reported range"
- If result is below `20%`: open issue to investigate
- If result is above `30%`: headline result confirmed — write it up

**Deliverable:** PR `paper-replication-1b` with results + updated README.

---

## Phase 4 — Final polish (1 day, $0)

- Tag `v1.0.0` with release notes summarizing the journey
- Optional 60-second demo GIF
- Security check: no secrets, no API keys in repo
- Open issues for any remaining known limitations
- Update `docs/changes-2026-05-01.md`

---

## Current status

- Phase 1 items 1.1–1.7: complete locally
- Phase 1 item 1.8: pending full-suite verification and PR
- Phase 2: complete locally, pending verification
- Phase 3–4: pending later decision/verification

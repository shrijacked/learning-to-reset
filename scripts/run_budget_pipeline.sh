#!/usr/bin/env bash
# One-shot replication via scripts/replicate_paper.sh with **this file’s** defaults
# (slightly larger budget + LTR_MAX_NEW_TOKENS default 768). Does not change replicate_paper.sh.
#
# Usage:
#   bash scripts/run_budget_pipeline.sh
#
# All knobs below can be overridden by exporting the same variable before invoking.
#
# --- Passed through to replicate_paper.sh as environment ---
# LTR_MAX_NEW_TOKENS          — max new tokens for steps 4 (raw eval), 7 (RLOO), 8 (final eval)
# LTR_SFT_MAX_TRAIN_EXAMPLES  — cap step-3 first SFT rows only
# LTR_EVAL_RAW_MAX_EXAMPLES   — cap step-4 raw eval prompts (mining pool); empty string = no cap
# LTR_MERGE_MAX_BASE_EXAMPLES — cap base rows in step-6 merge (after conversion)
# LTR_MERGE_MAX_MINED_EXAMPLES — cap mined rows in step-6 merge
# LTR_RLOO_MAX_TRAIN_EXAMPLES / LTR_RLOO_MAX_VALIDATION_EXAMPLES — cap Countdown rows loaded in RLOO
# LTR_EVAL_FINAL_MAX_EXAMPLES — cap step-8 prompts (unset = full hard split)
# LTR_VERIFIER_FEEDBACK       — set to 1 for step-8 --verifier-feedback (non-baseline)
#
# --- CLI flags to replicate_paper.sh (export before run) ---
# OUT_DIR, BASE_MODEL, SCALE, RLOO_STEPS, MAX_CLEAN_TRIES, RECOVERY_STYLE
#
# Other env vars you may set (not exported here; child Python processes inherit them):
#   LTR_FORCE_CPU=1  — force CPU in sft_runtime (see run.sh for typical use)
#   LTR_NO_TQDM=1    — disable tqdm in eval_runtime progress (optional)
#
# Not tunable via replicate_paper.sh env today: SFT --epochs (1), SFT --max-length (1024
# in sft_runtime defaults), second-SFT row cap (full combined JSONL), RLOO --temperature.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- defaults: “new budget” (vs setup.md manual 8k / 7.5k+500 / 384 tokens) ---
export OUT_DIR="${OUT_DIR:-$REPO_ROOT/runs/my-new-budget-run}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export SCALE="${SCALE:-paper}"
export RLOO_STEPS="${RLOO_STEPS:-100}"
export MAX_CLEAN_TRIES="${MAX_CLEAN_TRIES:-3}"
export RECOVERY_STYLE="${RECOVERY_STYLE:-grounded}"

export LTR_SFT_MAX_TRAIN_EXAMPLES="${LTR_SFT_MAX_TRAIN_EXAMPLES:-10000}"
export LTR_EVAL_RAW_MAX_EXAMPLES="${LTR_EVAL_RAW_MAX_EXAMPLES:-750}"
export LTR_MERGE_MAX_BASE_EXAMPLES="${LTR_MERGE_MAX_BASE_EXAMPLES:-10000}"
export LTR_MERGE_MAX_MINED_EXAMPLES="${LTR_MERGE_MAX_MINED_EXAMPLES:-750}"
export LTR_RLOO_MAX_TRAIN_EXAMPLES="${LTR_RLOO_MAX_TRAIN_EXAMPLES:-10000}"
export LTR_RLOO_MAX_VALIDATION_EXAMPLES="${LTR_RLOO_MAX_VALIDATION_EXAMPLES:-25000}"
export LTR_MAX_NEW_TOKENS="${LTR_MAX_NEW_TOKENS:-768}"

# Optional step-8 cap: export LTR_EVAL_FINAL_MAX_EXAMPLES=128 before running.
export LTR_VERIFIER_FEEDBACK="${LTR_VERIFIER_FEEDBACK:-0}"

echo "[run_budget_pipeline] OUT_DIR=$OUT_DIR BASE_MODEL=$BASE_MODEL SCALE=$SCALE"
echo "[run_budget_pipeline] RLOO_STEPS=$RLOO_STEPS LTR_MAX_NEW_TOKENS=$LTR_MAX_NEW_TOKENS MAX_CLEAN_TRIES=$MAX_CLEAN_TRIES"
echo "[run_budget_pipeline] LTR_SFT_MAX_TRAIN_EXAMPLES=$LTR_SFT_MAX_TRAIN_EXAMPLES LTR_EVAL_RAW_MAX_EXAMPLES=$LTR_EVAL_RAW_MAX_EXAMPLES"
echo "[run_budget_pipeline] LTR_MERGE_MAX_BASE_EXAMPLES=$LTR_MERGE_MAX_BASE_EXAMPLES LTR_MERGE_MAX_MINED_EXAMPLES=$LTR_MERGE_MAX_MINED_EXAMPLES"
echo "[run_budget_pipeline] LTR_RLOO_MAX_TRAIN_EXAMPLES=$LTR_RLOO_MAX_TRAIN_EXAMPLES LTR_RLOO_MAX_VALIDATION_EXAMPLES=$LTR_RLOO_MAX_VALIDATION_EXAMPLES"
if [[ -n "${LTR_EVAL_FINAL_MAX_EXAMPLES:-}" ]]; then
  echo "[run_budget_pipeline] LTR_EVAL_FINAL_MAX_EXAMPLES=$LTR_EVAL_FINAL_MAX_EXAMPLES"
fi

bash scripts/replicate_paper.sh \
  --base-model "$BASE_MODEL" \
  --scale-override "$SCALE" \
  --out-dir "$OUT_DIR" \
  --rloo-steps "$RLOO_STEPS" \
  --recovery-style "$RECOVERY_STYLE" \
  --max-clean-tries "$MAX_CLEAN_TRIES"

echo "[run_budget_pipeline] done. Main outputs: $OUT_DIR/eval-final/summary.json"

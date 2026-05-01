#!/usr/bin/env bash
# replicate_paper.sh - End-to-end paper replication pipeline.
#
# Drives the full Section 4 recipe from main.pdf:
#   1. Fetch paper-aligned source datasets       (paper_sources --scale paper)
#   2. Prepare SFT + Countdown artifacts         (prepare_paper_artifacts --scale paper)
#   3. SFT on grounded recovery traces           (sft_runtime)
#   4. Baseline raw eval on hard-mine raw prompts (eval_runtime --raw-generation)
#   5. Mine recovery traces from hard-mine       (failure_recovery_traces --exclude-source-ids)
#   6. Re-SFT on combined corpus                 (sft_runtime)
#   7. Reset-aware RLOO                          (rloo_runtime)
#   8. Final reset-aware eval (multi-clean)      (eval_runtime --max-clean-tries 3)
#   9. Extension comparison                      (run_extension_comparison.py)
#   10. Qualitative sample export                (export_qualitative_samples.py)
#
# Optional environment:
#   LTR_VERIFIER_FEEDBACK=1  — append decode-time verifier hints on step 8 retries
#                              (passes --verifier-feedback to eval_runtime; not in
#                              the original paper baseline, see README).
#
# Usage:
#   bash scripts/replicate_paper.sh --base-model Qwen/Qwen2.5-1.5B-Instruct \
#                                   --out-dir runs/replicate-paper-2026-04-26
#
# Dry run (parse every CLI without executing model code):
#   bash scripts/replicate_paper.sh --dry-run --base-model Qwen/Qwen2.5-0.5B \
#                                   --out-dir /tmp/replicate-paper-dryrun
#
# Hardware:
#   - Local CPU/MPS: only supported with Qwen 0.5B and tiny scale (use --scale-override pilot).
#   - Paper-faithful: requires a 1B+ GPU (see docs/paper-replication.md).
#
# Exit codes:
#   0 success
#   1 missing dependency or bad arg
#   2 a pipeline step failed
set -euo pipefail

PYTHON="${PYTHON:-python3}"
DRY_RUN=0
BASE_MODEL=""
OUT_DIR=""
SCALE="paper"
RECOVERY_STYLE="grounded"
MAX_CLEAN_TRIES=3
RLOO_STEPS=200

print_usage() {
    sed -n '2,30p' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --scale-override)
            SCALE="$2"
            shift 2
            ;;
        --recovery-style)
            RECOVERY_STYLE="$2"
            shift 2
            ;;
        --max-clean-tries)
            MAX_CLEAN_TRIES="$2"
            shift 2
            ;;
        --rloo-steps)
            RLOO_STEPS="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            print_usage
            exit 1
            ;;
    esac
done

if [[ -z "$BASE_MODEL" ]]; then
    echo "ERROR: --base-model is required (e.g. Qwen/Qwen2.5-1.5B-Instruct)." >&2
    exit 1
fi

if [[ -z "$OUT_DIR" ]]; then
    echo "ERROR: --out-dir is required (will be created if missing)." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
SOURCES_DIR="$OUT_DIR/sources"
ARTIFACTS_DIR="$OUT_DIR/artifacts"
SFT_DIR="$OUT_DIR/sft"
SFT2_DIR="$OUT_DIR/sft-mined"
RLOO_DIR="$OUT_DIR/rloo"
EVAL_RAW_DIR="$OUT_DIR/eval-raw"
EVAL_FINAL_DIR="$OUT_DIR/eval-final"
EXTENSIONS_DIR="$OUT_DIR/extensions"
QUALITATIVE_PATH="$OUT_DIR/qualitative.md"
MINED_TRACES_PATH="$OUT_DIR/mined-recovery-traces.jsonl"

# Use a string (not an empty bash array) so `set -u` on macOS /bin/bash does not
# treat "${ARRAY[@]}" as an unbound when the array is empty.
VERIFIER_FEEDBACK_FLAG=""
if [[ "${LTR_VERIFIER_FEEDBACK:-0}" == "1" ]]; then
    VERIFIER_FEEDBACK_FLAG="--verifier-feedback"
fi

log() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[replicate_paper][dry-run] $*"
    else
        echo "[replicate_paper] $*"
    fi
}

run_cmd() {
    log "step: $*"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        return 0
    fi
    "$@"
}

# Verifies a CLI parses by invoking `--help`. In real mode this is a no-op
# beyond a sanity check before each heavy step.
check_cli() {
    local label="$1"; shift
    if ! "$@" --help >/dev/null 2>&1; then
        echo "ERROR: CLI parse check failed for $label ($*)" >&2
        exit 2
    fi
    log "cli ok: $label"
}

###############################################################################
# Step 1: Fetch paper sources (full scale).
###############################################################################
check_cli "paper_sources" "$PYTHON" -m learning_to_reset.paper_sources
run_cmd "$PYTHON" -m learning_to_reset.paper_sources \
    --scale "$SCALE" \
    --output-dir "$SOURCES_DIR"

###############################################################################
# Step 2: Prepare SFT + Countdown artifacts (paper recipe).
###############################################################################
check_cli "prepare_paper_artifacts" "$PYTHON" -m learning_to_reset.prepare_paper_artifacts
run_cmd "$PYTHON" -m learning_to_reset.prepare_paper_artifacts \
    --scale "$SCALE" \
    --traces "$SOURCES_DIR/traces.jsonl" \
    --countdown-train "$SOURCES_DIR/countdown-train.jsonl" \
    --countdown-eval "$SOURCES_DIR/countdown-eval.jsonl" \
    --output-dir "$ARTIFACTS_DIR"

###############################################################################
# Step 3: First SFT pass on grounded recovery traces.
###############################################################################
check_cli "sft_runtime" "$PYTHON" -m learning_to_reset.sft_runtime
run_cmd "$PYTHON" -m learning_to_reset.sft_runtime \
    --train "$ARTIFACTS_DIR/sft-train.jsonl" \
    --validation "$ARTIFACTS_DIR/sft-validation.jsonl" \
    --model "$BASE_MODEL" \
    --output-dir "$SFT_DIR" \
    --epochs 1.0

###############################################################################
# Step 4: Baseline raw eval (used to mine failures).
###############################################################################
check_cli "eval_runtime" "$PYTHON" -m learning_to_reset.eval_runtime
run_cmd "$PYTHON" -m learning_to_reset.eval_runtime \
    --prepared-countdown "$ARTIFACTS_DIR/countdown-mine-hard-raw.jsonl" \
    --model "$SFT_DIR" \
    --output-dir "$EVAL_RAW_DIR" \
    --max-new-tokens 384 \
    --raw-generation

###############################################################################
# Step 5: Mine failures into recovery traces, contamination-guarded.
###############################################################################
check_cli "failure_recovery_traces" "$PYTHON" -m learning_to_reset.failure_recovery_traces
run_cmd "$PYTHON" -m learning_to_reset.failure_recovery_traces \
    --prepared-countdown "$ARTIFACTS_DIR/countdown-mine-hard.jsonl" \
    --eval-results "$EVAL_RAW_DIR/results.jsonl" \
    --output-path "$MINED_TRACES_PATH" \
    --recovery-style "$RECOVERY_STYLE" \
    --exclude-source-ids "$ARTIFACTS_DIR/countdown-test.jsonl" \
    --exclude-source-ids "$ARTIFACTS_DIR/countdown-test-hard.jsonl"

###############################################################################
# Step 6: Re-SFT on the combined corpus.
###############################################################################
if [[ "$DRY_RUN" -eq 0 ]]; then
    cat "$ARTIFACTS_DIR/sft-train.jsonl" "$MINED_TRACES_PATH" \
        > "$OUT_DIR/sft-train-combined.jsonl"
fi
run_cmd "$PYTHON" -m learning_to_reset.sft_runtime \
    --train "$OUT_DIR/sft-train-combined.jsonl" \
    --validation "$ARTIFACTS_DIR/sft-validation.jsonl" \
    --model "$BASE_MODEL" \
    --output-dir "$SFT2_DIR" \
    --epochs 1.0

###############################################################################
# Step 7: Reset-aware RLOO.
###############################################################################
check_cli "rloo_runtime" "$PYTHON" -m learning_to_reset.rloo_runtime
run_cmd "$PYTHON" -m learning_to_reset.rloo_runtime \
    --train "$ARTIFACTS_DIR/countdown-train.jsonl" \
    --validation "$ARTIFACTS_DIR/countdown-validation.jsonl" \
    --model "$SFT2_DIR" \
    --output-dir "$RLOO_DIR" \
    --steps "$RLOO_STEPS" \
    --responses-per-prompt 4 \
    --max-new-tokens 384 \
    --temperature 1.0

###############################################################################
# Step 8: Final reset-aware eval with multi-clean decoding.
###############################################################################
run_cmd "$PYTHON" -m learning_to_reset.eval_runtime \
    --prepared-countdown "$ARTIFACTS_DIR/countdown-test-hard.jsonl" \
    --model "$RLOO_DIR" \
    --output-dir "$EVAL_FINAL_DIR" \
    --max-new-tokens 384 \
    --max-clean-tries "$MAX_CLEAN_TRIES" \
    $VERIFIER_FEEDBACK_FLAG

###############################################################################
# Step 9: Extension comparison (full-reset / retention / memory).
###############################################################################
EXTENSION_SCRIPT="scripts/run_extension_comparison.py"
if [[ -f "$EXTENSION_SCRIPT" ]]; then
    check_cli "run_extension_comparison" "$PYTHON" "$EXTENSION_SCRIPT"
    run_cmd "$PYTHON" "$EXTENSION_SCRIPT" \
        --prepared-countdown "$ARTIFACTS_DIR/countdown-test-hard.jsonl" \
        --results-jsonl "$EVAL_FINAL_DIR/results.jsonl" \
        --output-dir "$EXTENSIONS_DIR" \
        --max-cleans "$MAX_CLEAN_TRIES"
else
    log "skip step 9: $EXTENSION_SCRIPT not yet present (added in B3)."
fi

###############################################################################
# Step 10: Qualitative sample export (Figure 7).
###############################################################################
QUAL_SCRIPT="scripts/export_qualitative_samples.py"
if [[ -f "$QUAL_SCRIPT" ]]; then
    check_cli "export_qualitative_samples" "$PYTHON" "$QUAL_SCRIPT"
    run_cmd "$PYTHON" "$QUAL_SCRIPT" \
        --eval-results "$EVAL_FINAL_DIR/results.jsonl" \
        --output-path "$QUALITATIVE_PATH"
else
    log "skip step 10: $QUAL_SCRIPT not present."
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run complete. Every CLI parsed and every output path was planned."
    exit 0
fi

log "replication run complete. Outputs in $OUT_DIR"

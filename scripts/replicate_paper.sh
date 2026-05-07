#!/usr/bin/env bash
# replicate_paper.sh - End-to-end paper replication pipeline.
#
# Drives the full Section 4 recipe from main.pdf:
#   1. Fetch paper-aligned source datasets       (paper_sources --scale paper)
#   2. Prepare SFT + Countdown artifacts         (prepare_paper_artifacts --scale paper)
#   3. SFT on grounded recovery traces           (sft_runtime)
#   4. Baseline raw eval to mine failures        (eval_runtime --raw-generation;
#                                                 use countdown-train, not test-hard,
#                                                 so step 5 mining passes contamination check)
#   5. Mine recovery traces from failures        (failure_recovery_traces --exclude-source-ids)
#   6. Re-SFT on combined corpus                 (sft_runtime)
#   7. Reset-aware RLOO                          (rloo_runtime)
#   8. Final reset-aware eval (multi-clean)      (eval_runtime --max-clean-tries 3)
#   9. Extension comparison                      (run_extension_comparison.py)
#   10. Qualitative sample export                (export_qualitative_samples.py)
#
# Optional environment:
#   LTR_MAX_NEW_TOKENS — generation budget for steps 4, 7, and 8 (default 384).
#   LTR_EVAL_RAW_MAX_EXAMPLES — cap step 4 raw eval prompts (default 500 when unset).
#                              Export empty (LTR_EVAL_RAW_MAX_EXAMPLES=) for full file.
#   LTR_SFT_MAX_TRAIN_EXAMPLES — cap rows for step 3 SFT only (e.g. 10000 for a faster run).
#                               Does not apply to step 6: combined.jsonl is sft-train then mined;
#                               capping there would drop mined traces. Unset = full sft-train.jsonl.
#   LTR_RLOO_MAX_TRAIN_EXAMPLES / LTR_RLOO_MAX_VALIDATION_EXAMPLES — cap JSONL rows loaded in
#                               step 7 (RLOO does not use the full train file per step, but loading
#                               it still parses every line unless capped).
#   LTR_EVAL_FINAL_MAX_EXAMPLES — cap step 8 eval prompts (eval_runtime --max-examples).
#   LTR_VERIFIER_FEEDBACK=1  — append decode-time verifier hints on step 8 retries
#                              (passes --verifier-feedback to eval_runtime; not in
#                              the original paper baseline, see README).
#
# SFT source controls:
#   --sft-source-mode {reference,synthetic-countdown,mixed}
#   --sft-mix-reference-ratio FLOAT
#   --synthetic-max-samples N
#   --synthetic-solutions-per-sample N
#   --synthetic-recovery-style {walkthrough,verification,contrastive,grounded,both,all}
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

# Resolve repo root so this script works no matter the caller's cwd, and so
# scripts/*.py paths and `python -m learning_to_reset` imports are deterministic.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || {
    echo "ERROR: cannot cd to repo root ${REPO_ROOT}" >&2
    exit 1
}
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PYTHON="${PYTHON:-python3}"
DRY_RUN=0
BASE_MODEL=""
OUT_DIR=""
SCALE="paper"
RECOVERY_STYLE="grounded"
MAX_CLEAN_TRIES=3
RLOO_STEPS=200
SFT_SOURCE_MODE="synthetic-countdown"
SFT_MIX_REFERENCE_RATIO="0.0"
SYNTHETIC_MAX_SAMPLES=""
SYNTHETIC_SOLUTIONS_PER_SAMPLE=1
SYNTHETIC_RECOVERY_STYLE="grounded"

print_usage() {
    sed -n '2,45p' "$0"
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
        --sft-source-mode)
            SFT_SOURCE_MODE="$2"
            shift 2
            ;;
        --sft-mix-reference-ratio)
            SFT_MIX_REFERENCE_RATIO="$2"
            shift 2
            ;;
        --synthetic-max-samples)
            SYNTHETIC_MAX_SAMPLES="$2"
            shift 2
            ;;
        --synthetic-solutions-per-sample)
            SYNTHETIC_SOLUTIONS_PER_SAMPLE="$2"
            shift 2
            ;;
        --synthetic-recovery-style)
            SYNTHETIC_RECOVERY_STYLE="$2"
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

case "$SFT_SOURCE_MODE" in
    reference|synthetic-countdown|mixed)
        ;;
    *)
        echo "ERROR: --sft-source-mode must be one of reference, synthetic-countdown, mixed." >&2
        exit 1
        ;;
esac

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
REFERENCE_TRACES_PATH="$SOURCES_DIR/reference-traces.jsonl"
SYNTHETIC_TRACES_PATH="$SOURCES_DIR/sft-synthetic-traces.jsonl"
MIXED_TRACES_PATH="$SOURCES_DIR/sft-mixed-traces.jsonl"
SELECTED_TRACES_PATH="$REFERENCE_TRACES_PATH"

# Optional step-8 flag (avoid empty-array + set -u pitfalls).
LTR_VERIFIER_FEEDBACK="${LTR_VERIFIER_FEEDBACK:-0}"

# Step 4: default 500 prompts for raw eval. Export LTR_EVAL_RAW_MAX_EXAMPLES= (empty) for no cap.
if [[ "${LTR_EVAL_RAW_MAX_EXAMPLES-unset}" == "unset" ]]; then
    LTR_EVAL_RAW_MAX_EXAMPLES=500
fi

LTR_MAX_NEW_TOKENS="${LTR_MAX_NEW_TOKENS:-384}"

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
# Step 1.5: Generate/select SFT trace source.
###############################################################################
case "$SFT_SOURCE_MODE" in
    reference)
        SELECTED_TRACES_PATH="$REFERENCE_TRACES_PATH"
        ;;
    synthetic-countdown|mixed)
        check_cli "synthetic_countdown_traces" "$PYTHON" -m learning_to_reset.synthetic_countdown_traces
        SYNTHETIC_CMD=(
            -m learning_to_reset.synthetic_countdown_traces
            --countdown "$SOURCES_DIR/countdown-train.jsonl"
            --output-path "$SYNTHETIC_TRACES_PATH"
            --solutions-per-sample "$SYNTHETIC_SOLUTIONS_PER_SAMPLE"
            --recovery-style "$SYNTHETIC_RECOVERY_STYLE"
        )
        if [[ -n "$SYNTHETIC_MAX_SAMPLES" ]]; then
            SYNTHETIC_CMD+=(--max-samples "$SYNTHETIC_MAX_SAMPLES")
        fi
        run_cmd "$PYTHON" "${SYNTHETIC_CMD[@]}"
        SELECTED_TRACES_PATH="$SYNTHETIC_TRACES_PATH"
        if [[ "$SFT_SOURCE_MODE" == "mixed" ]]; then
            check_cli "mix_sft_trace_sources" "$PYTHON" -m learning_to_reset.mix_sft_trace_sources
            run_cmd "$PYTHON" -m learning_to_reset.mix_sft_trace_sources \
                --synthetic "$SYNTHETIC_TRACES_PATH" \
                --reference "$REFERENCE_TRACES_PATH" \
                --output-path "$MIXED_TRACES_PATH" \
                --reference-ratio "$SFT_MIX_REFERENCE_RATIO"
            SELECTED_TRACES_PATH="$MIXED_TRACES_PATH"
        fi
        ;;
esac
log "SFT source mode: ${SFT_SOURCE_MODE} (${SELECTED_TRACES_PATH})"

###############################################################################
# Step 2: Prepare SFT + Countdown artifacts (paper recipe).
###############################################################################
check_cli "prepare_paper_artifacts" "$PYTHON" -m learning_to_reset.prepare_paper_artifacts
run_cmd "$PYTHON" -m learning_to_reset.prepare_paper_artifacts \
    --scale "$SCALE" \
    --traces "$SELECTED_TRACES_PATH" \
    --trace-source-mode "$SFT_SOURCE_MODE" \
    --countdown-train "$SOURCES_DIR/countdown-train.jsonl" \
    --countdown-eval "$SOURCES_DIR/countdown-eval.jsonl" \
    --output-dir "$ARTIFACTS_DIR"

check_cli "sft_corpus_inspect" "$PYTHON" -m learning_to_reset.sft_corpus_inspect
run_cmd "$PYTHON" -m learning_to_reset.sft_corpus_inspect \
    --path "$ARTIFACTS_DIR/sft-train.jsonl" \
    --path "$ARTIFACTS_DIR/sft-validation.jsonl" \
    --output "$ARTIFACTS_DIR/sft-composition-report.json"

###############################################################################
# Step 3: First SFT pass on grounded recovery traces.
###############################################################################
check_cli "sft_runtime" "$PYTHON" -m learning_to_reset.sft_runtime
STEP3_SFT=(
    -m learning_to_reset.sft_runtime
    --train "$ARTIFACTS_DIR/sft-train.jsonl"
    --validation "$ARTIFACTS_DIR/sft-validation.jsonl"
    --model "$BASE_MODEL"
    --output-dir "$SFT_DIR"
    --epochs 1.0
    --strict-input-schema
)
if [[ -n "${LTR_SFT_MAX_TRAIN_EXAMPLES:-}" ]]; then
    STEP3_SFT+=(--max-train-examples "${LTR_SFT_MAX_TRAIN_EXAMPLES}")
fi
run_cmd "$PYTHON" "${STEP3_SFT[@]}"

###############################################################################
# Step 4: Baseline raw eval (used to mine failures).
# Must match train-side prompts — eval on test-hard makes every source_id overlap
# step 5's --exclude-source-ids and mining aborts.
###############################################################################
check_cli "eval_runtime" "$PYTHON" -m learning_to_reset.eval_runtime
STEP4_RAW=(
    -m learning_to_reset.eval_runtime
    --prepared-countdown "$ARTIFACTS_DIR/countdown-train.jsonl"
    --model "$SFT_DIR"
    --output-dir "$EVAL_RAW_DIR"
    --max-new-tokens "${LTR_MAX_NEW_TOKENS}"
    --raw-generation
)
if [[ -n "${LTR_EVAL_RAW_MAX_EXAMPLES}" ]]; then
    STEP4_RAW+=(--max-examples "${LTR_EVAL_RAW_MAX_EXAMPLES}")
fi
run_cmd "$PYTHON" "${STEP4_RAW[@]}"

###############################################################################
# Step 5: Mine failures into recovery traces, contamination-guarded.
###############################################################################
check_cli "failure_recovery_traces" "$PYTHON" -m learning_to_reset.failure_recovery_traces
run_cmd "$PYTHON" -m learning_to_reset.failure_recovery_traces \
    --prepared-countdown "$ARTIFACTS_DIR/countdown-train.jsonl" \
    --eval-results "$EVAL_RAW_DIR/results.jsonl" \
    --output-path "$MINED_TRACES_PATH" \
    --recovery-style "$RECOVERY_STYLE" \
    --exclude-source-ids "$ARTIFACTS_DIR/countdown-test.jsonl" \
    --exclude-source-ids "$ARTIFACTS_DIR/countdown-test-hard.jsonl"

###############################################################################
# Step 6: Re-SFT on the combined corpus.
# Mined JSONL is TraceRecord-shaped; base SFT is PromptExample-shaped — do not cat.
###############################################################################
check_cli "merge_sft_corpus" "$PYTHON" -m learning_to_reset.merge_sft_corpus
STEP6_MERGE=(
    -m learning_to_reset.merge_sft_corpus
    --sft-train "$ARTIFACTS_DIR/sft-train.jsonl"
    --mined-traces "$MINED_TRACES_PATH"
    --output "$OUT_DIR/sft-train-combined.jsonl"
)
if [[ -n "${LTR_MERGE_MAX_BASE_EXAMPLES:-}" ]]; then
    STEP6_MERGE+=(--max-base-examples "${LTR_MERGE_MAX_BASE_EXAMPLES}")
fi
if [[ -n "${LTR_MERGE_MAX_MINED_EXAMPLES:-}" ]]; then
    STEP6_MERGE+=(--max-mined-examples "${LTR_MERGE_MAX_MINED_EXAMPLES}")
fi
run_cmd "$PYTHON" "${STEP6_MERGE[@]}"
run_cmd "$PYTHON" -m learning_to_reset.sft_corpus_inspect \
    --path "$OUT_DIR/sft-train-combined.jsonl" \
    --path "$ARTIFACTS_DIR/sft-validation.jsonl" \
    --output "$OUT_DIR/sft-composition-combined-report.json"
run_cmd "$PYTHON" -m learning_to_reset.sft_runtime \
    --train "$OUT_DIR/sft-train-combined.jsonl" \
    --validation "$ARTIFACTS_DIR/sft-validation.jsonl" \
    --model "$BASE_MODEL" \
    --output-dir "$SFT2_DIR" \
    --epochs 1.0 \
    --strict-input-schema

###############################################################################
# Step 7: Reset-aware RLOO.
###############################################################################
check_cli "rloo_runtime" "$PYTHON" -m learning_to_reset.rloo_runtime
STEP7_RLOO=(
    -m learning_to_reset.rloo_runtime
    --train "$ARTIFACTS_DIR/countdown-train.jsonl"
    --validation "$ARTIFACTS_DIR/countdown-validation.jsonl"
    --model "$SFT2_DIR"
    --output-dir "$RLOO_DIR"
    --steps "$RLOO_STEPS"
    --responses-per-prompt 4
    --max-new-tokens "${LTR_MAX_NEW_TOKENS}"
    --temperature 1.0
)
if [[ -n "${LTR_RLOO_MAX_TRAIN_EXAMPLES:-}" ]]; then
    STEP7_RLOO+=(--max-train-examples "${LTR_RLOO_MAX_TRAIN_EXAMPLES}")
fi
if [[ -n "${LTR_RLOO_MAX_VALIDATION_EXAMPLES:-}" ]]; then
    STEP7_RLOO+=(--max-validation-examples "${LTR_RLOO_MAX_VALIDATION_EXAMPLES}")
fi
run_cmd "$PYTHON" "${STEP7_RLOO[@]}"

###############################################################################
# Step 8: Final reset-aware eval with multi-clean decoding.
###############################################################################
STEP8_EVAL=(
    -m learning_to_reset.eval_runtime
    --prepared-countdown "$ARTIFACTS_DIR/countdown-test-hard.jsonl"
    --model "$RLOO_DIR"
    --output-dir "$EVAL_FINAL_DIR"
    --max-new-tokens "${LTR_MAX_NEW_TOKENS}"
    --max-clean-tries "$MAX_CLEAN_TRIES"
)
if [[ "$LTR_VERIFIER_FEEDBACK" == "1" ]]; then
    STEP8_EVAL+=(--verifier-feedback)
fi
if [[ -n "${LTR_EVAL_FINAL_MAX_EXAMPLES:-}" ]]; then
    STEP8_EVAL+=(--max-examples "${LTR_EVAL_FINAL_MAX_EXAMPLES}")
fi
run_cmd "$PYTHON" "${STEP8_EVAL[@]}"

###############################################################################
# Step 9: Extension comparison (full-reset / retention / memory).
###############################################################################
EXTENSION_SCRIPT="$REPO_ROOT/scripts/run_extension_comparison.py"
if [[ -f "$EXTENSION_SCRIPT" ]]; then
    check_cli "run_extension_comparison" "$PYTHON" "$EXTENSION_SCRIPT"
    run_cmd "$PYTHON" "$EXTENSION_SCRIPT" \
        --prepared-countdown "$ARTIFACTS_DIR/countdown-test-hard.jsonl" \
        --results-jsonl "$EVAL_FINAL_DIR/results.jsonl" \
        --output-dir "$EXTENSIONS_DIR" \
        --max-cleans "$MAX_CLEAN_TRIES"
else
    log "skip step 9: ${EXTENSION_SCRIPT} not present."
fi

###############################################################################
# Step 10: Qualitative sample export (Figure 7).
###############################################################################
QUAL_SCRIPT="$REPO_ROOT/scripts/export_qualitative_samples.py"
if [[ -f "$QUAL_SCRIPT" ]]; then
    check_cli "export_qualitative_samples" "$PYTHON" "$QUAL_SCRIPT"
    run_cmd "$PYTHON" "$QUAL_SCRIPT" \
        --eval-results "$EVAL_FINAL_DIR/results.jsonl" \
        --output-path "$QUALITATIVE_PATH"
else
    log "skip step 10: ${QUAL_SCRIPT} not present."
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run complete. Every CLI parsed and every output path was planned."
    exit 0
fi

log "replication run complete. Outputs in $OUT_DIR"

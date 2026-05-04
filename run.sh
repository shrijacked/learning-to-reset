#!/usr/bin/env bash
# End-to-end entrypoint for course / CI evaluation (Ubuntu 22.04+, bash only).
# Clones the repo, then from repo root:  bash run.sh
#
# Creates a venv, installs Python deps, runs scripts/replicate_paper.sh (full pipeline),
# builds figures, and copies report pointers under outputs/.
#
# Environment (all optional):
#   RUN_FULL=1              — paper data scale + 1.5B-style base model (needs GPU + time + HF egress).
#   RUN_TESTS=1             — run unit tests before pipeline (may fail on minimal environments).
#   SKIP_NETWORK_CHECK=1    — skip Hugging Face connectivity probe.
#   HF_TOKEN                — forwarded as HUGGING_FACE_HUB_TOKEN for gated models.
#   LTR_FORCE_CPU=1         — force CPU training/eval (default for RUN_FULL=0 / pilot).
#   LTR_RLOO_STEPS=N        — RLOO optimization steps (defaults below).
#   LTR_EVAL_RAW_MAX_EXAMPLES / LTR_SFT_MAX_TRAIN_EXAMPLES / LTR_MERGE_* / LTR_RLOO_MAX_* /
#   LTR_EVAL_FINAL_MAX_EXAMPLES — forwarded to replicate_paper.sh (see that script’s header).
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

mkdir -p outputs/logs
exec > >(tee -a outputs/logs/run.log) 2>&1

echo "=========================================="
echo "[run.sh] Learning to Reset — automated run"
echo "[run.sh] REPO_ROOT=${REPO_ROOT}"
echo "=========================================="

echo "[run.sh] (1/6) System bootstrap (python3, venv packages on Debian if root)"
bash scripts/bootstrap_ubuntu.sh

if ! command -v python3 >/dev/null 2>&1; then
  echo "[run.sh] ERROR: python3 is required. On Ubuntu: sudo apt-get install -y python3 python3-venv python3-pip" >&2
  exit 1
fi

# Ensure venv module works (ubuntu minimal images need python3-venv)
if ! python3 -m venv --help >/dev/null 2>&1; then
  echo "[run.sh] ERROR: 'python3 -m venv' failed. Install: sudo apt-get install -y python3-venv" >&2
  exit 1
fi

echo "[run.sh] (2/6) Create and activate virtual environment"
rm -rf .venv
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo "[run.sh] (3/6) Install Python dependencies (relative paths only)"
python -m pip install -r requirements.txt
python -m pip install -e .

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HF_HOME="${REPO_ROOT}/.hf-cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"
export TOKENIZERS_PARALLELISM=false

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

if [[ "${SKIP_NETWORK_CHECK:-0}" != "1" ]]; then
  echo "[run.sh] checking HTTPS egress to huggingface.co (set SKIP_NETWORK_CHECK=1 to skip)"
  if ! python3 -c "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=20)"; then
    echo "[run.sh] ERROR: cannot reach https://huggingface.co — model/dataset download will fail." >&2
    exit 1
  fi
fi

if [[ "${RUN_TESTS:-0}" == "1" ]]; then
  echo "[run.sh] (optional) unit tests"
  python3 -m unittest discover -s tests -q
fi

# Default: fast CPU-friendly pilot; set RUN_FULL=1 for full paper recipe (long-running).
if [[ "${RUN_FULL:-0}" == "1" ]]; then
  echo "[run.sh] (4/6) Full replication (paper scale, long-running)"
  export LTR_FORCE_CPU="${LTR_FORCE_CPU:-0}"
  BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
  SCALE_OVERRIDE="paper"
  export LTR_RLOO_STEPS="${LTR_RLOO_STEPS:-50}"
else
  echo "[run.sh] (4/6) Pilot replication (small data, CPU-forced by default — suitable for grading containers)"
  export LTR_FORCE_CPU="${LTR_FORCE_CPU:-1}"
  BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B}"
  SCALE_OVERRIDE="pilot"
  # Reasonable caps for wall-clock; override via env if needed.
  export LTR_EVAL_RAW_MAX_EXAMPLES="${LTR_EVAL_RAW_MAX_EXAMPLES:-256}"
  export LTR_RLOO_MAX_TRAIN_EXAMPLES="${LTR_RLOO_MAX_TRAIN_EXAMPLES:-2048}"
  export LTR_RLOO_MAX_VALIDATION_EXAMPLES="${LTR_RLOO_MAX_VALIDATION_EXAMPLES:-512}"
  export LTR_EVAL_FINAL_MAX_EXAMPLES="${LTR_EVAL_FINAL_MAX_EXAMPLES:-128}"
  export LTR_RLOO_STEPS="${LTR_RLOO_STEPS:-15}"
fi

RLOO_STEPS="${LTR_RLOO_STEPS}"

OUT_DIR="outputs/replicate-paper"
mkdir -p "$OUT_DIR"

echo "[run.sh] BASE_MODEL=${BASE_MODEL} SCALE_OVERRIDE=${SCALE_OVERRIDE} RLOO_STEPS=${RLOO_STEPS}"
echo "[run.sh] OUT_DIR=${OUT_DIR} (relative)"

bash scripts/replicate_paper.sh \
  --base-model "$BASE_MODEL" \
  --scale-override "$SCALE_OVERRIDE" \
  --rloo-steps "$RLOO_STEPS" \
  --out-dir "$OUT_DIR"

echo "[run.sh] (5/6) Build figures (paper + budget replicate tables)"
PYTHONPATH="${REPO_ROOT}/src" python3 scripts/build_figures.py \
  --output-dir outputs/figures \
  --budget-dir outputs/figures/budget

echo "[run.sh] (6/6) Collect report artifacts under outputs/report/"
mkdir -p outputs/report
for f in pre-readme.md README.md result.md; do
  if [[ -f "$f" ]]; then
    cp -f "$f" "outputs/report/"
  fi
done

{
  echo "Primary run directory: outputs/replicate-paper/"
  echo "  sources/   — fetched HF datasets"
  echo "  artifacts/ — prepared JSONL"
  echo "  sft/ sft-mined/ rloo/ — checkpoints"
  echo "  eval-raw/ eval-final/ — evaluation JSON + summary.json"
  echo "  extensions/ qualitative.md — downstream exports"
  echo "Figures: outputs/figures/ and outputs/figures/budget/"
  echo "Full log: outputs/logs/run.log"
} > outputs/ARTIFACTS.txt

echo ""
echo "=========================================="
echo "[run.sh] SUCCESS"
echo "=========================================="
cat outputs/ARTIFACTS.txt
echo ""
echo "Key files:"
echo "  outputs/replicate-paper/eval-final/summary.json"
echo "  outputs/replicate-paper/eval-final/results.jsonl"
echo "  outputs/replicate-paper/qualitative.md"
echo "  outputs/figures/budget/milestone-leaderboard.svg"
echo "  outputs/logs/run.log"

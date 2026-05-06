# Setup — manual, step by step

Terminal only. This doc is for **manual** runs (no `run.sh`). You install deps yourself, log into Hugging Face, then run each pipeline step in order.

## One-time: environment

From the repo root:

```bash
cd /path/to/learning-to-reset
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

**Hugging Face:** after `huggingface-cli login`, downloads use your saved token. If a step still fails with auth errors, set `export HUGGING_FACE_HUB_TOKEN=hf_...` for that shell.

**GPU:** paper-scale training and eval expect CUDA. CPU-only is not realistic at this scale.

---

## Budget protocol (sample counts)

These are the **resource-bounded** caps aligned with the “8k first SFT → mine → **7.5k base + 500 mined** → second SFT → RLOO on capped Countdown pools” story:

| Stage | What is capped | Value |
|--------|----------------|--------|
| Step 3 — first SFT | Rows from `sft-train.jsonl` | **8000** (`--max-train-examples 8000`) |
| Step 4 — raw eval (mining) | Prompts decoded | **500** (`--max-examples 500`; matches `replicate_paper.sh` default) |
| Step 6a — merge | Rows from original SFT train / from mined traces | **7500** + **500** (`merge_sft_corpus` caps) → **8000** lines in `sft-train-combined.jsonl` |
| Step 6b — second SFT | Training file | **all lines** in combined file (8000 if merge caps as above); no extra cap in the orchestrator |
| Step 7 — RLOO | Countdown train / val JSONL rows loaded | **8000** / **25000** (optional caps; see note below) |
| Step 7 — RLOO | Optimization steps | **100** here (tune with `RLOO_STEPS` / `--steps`) |
| Step 8 — final eval | Hard split | **full** `countdown-test-hard.jsonl` unless you set `--max-examples N` for a smoke test |

**RLOO train/val caps.** At paper scale, `countdown-train.jsonl` can be **very** large (on the order of hundreds of thousands of lines). The **8000 / 25000** limits are **not** “using a huge dataset” in absolute terms — they **trim** what gets loaded into RAM and parsed each run so RLOO stays tractable. Compared to the full file they are a **subset**; compared to a tiny pilot (e.g. a few thousand train rows) they are **moderate**. You can **lower** both for faster iteration (noisier validation, less diverse batches) or **drop** `--max-train-examples` / `--max-validation-examples` entirely if you want the full prepared JSONLs (slow and memory-heavy).

Do **not** use `--max-train-examples 8000` on the **combined** file to approximate “7.5k + 500”: that flag takes the **first** N lines only, which are all **base** rows — mined traces would never appear. Use `merge_sft_corpus` with `--max-base-examples` / `--max-mined-examples` instead.

---

## Paths and model (set once per run)

Run everything from the **repository root** with your venv activated.

```bash
export OUT_DIR="$PWD/runs/my-budget-run"   # change as you like
export BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
export SCALE="paper"                       # full data fetch + paper prep recipe
export RECOVERY_STYLE="grounded"
export RLOO_STEPS="100"
# Optional: tighten RLOO data caps (defaults in Step 7 block below)
export RLOO_MAX_TRAIN="8000"
export RLOO_MAX_VAL="25000"
export MAX_CLEAN_TRIES="3"

mkdir -p "$OUT_DIR"
SOURCES_DIR="$OUT_DIR/sources"
ARTIFACTS_DIR="$OUT_DIR/artifacts"
SFT_DIR="$OUT_DIR/sft"
SFT2_DIR="$OUT_DIR/sft-mined"
RLOO_DIR="$OUT_DIR/rloo"
EVAL_RAW_DIR="$OUT_DIR/eval-raw"
EVAL_FINAL_DIR="$OUT_DIR/eval-final"
EXTENSIONS_DIR="$OUT_DIR/extensions"
MINED_TRACES_PATH="$OUT_DIR/mined-recovery-traces.jsonl"
COMBINED_SFT="$OUT_DIR/sft-train-combined.jsonl"
QUALITATIVE_PATH="$OUT_DIR/qualitative.md"
```

The commands below use `python3 -m ...` after `pip install -e .` (same as `scripts/replicate_paper.sh`).

---

## Step 1 — Fetch paper sources

```bash
python3 -m learning_to_reset.paper_sources \
  --scale "$SCALE" \
  --output-dir "$SOURCES_DIR"
```

---

## Step 2 — Prepare SFT + Countdown artifacts

```bash
python3 -m learning_to_reset.prepare_paper_artifacts \
  --scale "$SCALE" \
  --traces "$SOURCES_DIR/reference-traces.jsonl" \
  --countdown-train "$SOURCES_DIR/countdown-train.jsonl" \
  --countdown-eval "$SOURCES_DIR/countdown-eval.jsonl" \
  --output-dir "$ARTIFACTS_DIR"
```

---

## Step 3 — First SFT (cap: **8000** train examples)

```bash
python3 -m learning_to_reset.sft_runtime \
  --train "$ARTIFACTS_DIR/sft-train.jsonl" \
  --validation "$ARTIFACTS_DIR/sft-validation.jsonl" \
  --model "$BASE_MODEL" \
  --output-dir "$SFT_DIR" \
  --epochs 1.0 \
  --max-train-examples 8000
```

---

## Step 4 — Raw generation eval (mining pool; cap: **500** prompts)

```bash
python3 -m learning_to_reset.eval_runtime \
  --prepared-countdown "$ARTIFACTS_DIR/countdown-train.jsonl" \
  --model "$SFT_DIR" \
  --output-dir "$EVAL_RAW_DIR" \
  --max-new-tokens 384 \
  --raw-generation \
  --max-examples 500
```

---

## Step 5 — Mine failures into recovery traces

```bash
python3 -m learning_to_reset.failure_recovery_traces \
  --prepared-countdown "$ARTIFACTS_DIR/countdown-train.jsonl" \
  --eval-results "$EVAL_RAW_DIR/results.jsonl" \
  --output-path "$MINED_TRACES_PATH" \
  --recovery-style "$RECOVERY_STYLE" \
  --exclude-source-ids "$ARTIFACTS_DIR/countdown-test.jsonl" \
  --exclude-source-ids "$ARTIFACTS_DIR/countdown-test-hard.jsonl"
```

---

## Step 6 — Merge (**7500** base + **500** mined) then second SFT

Merge (do not `cat` JSONLs — formats differ):

```bash
python3 -m learning_to_reset.merge_sft_corpus \
  --sft-train "$ARTIFACTS_DIR/sft-train.jsonl" \
  --mined-traces "$MINED_TRACES_PATH" \
  --output "$COMBINED_SFT" \
  --max-base-examples 7500 \
  --max-mined-examples 500
```

Second SFT on the combined file:

```bash
python3 -m learning_to_reset.sft_runtime \
  --train "$COMBINED_SFT" \
  --validation "$ARTIFACTS_DIR/sft-validation.jsonl" \
  --model "$BASE_MODEL" \
  --output-dir "$SFT2_DIR" \
  --epochs 1.0
```

---

## Step 7 — Reset-aware RLOO (train / val caps + **100** steps by default)

```bash
python3 -m learning_to_reset.rloo_runtime \
  --train "$ARTIFACTS_DIR/countdown-train.jsonl" \
  --validation "$ARTIFACTS_DIR/countdown-validation.jsonl" \
  --model "$SFT2_DIR" \
  --output-dir "$RLOO_DIR" \
  --steps "$RLOO_STEPS" \
  --responses-per-prompt 4 \
  --max-new-tokens 384 \
  --temperature 1.0 \
  --max-train-examples "${RLOO_MAX_TRAIN:-8000}" \
  --max-validation-examples "${RLOO_MAX_VAL:-25000}"
```

To use **less** data (faster, lighter): set `RLOO_MAX_TRAIN` / `RLOO_MAX_VAL` smaller before this step, or remove the two `--max-*` lines to load the **entire** prepared train/val files (only if you have the RAM and patience).

---

## Step 8 — Final eval on hard split (`max_clean_tries = 3`)

Full hard evaluation:

```bash
python3 -m learning_to_reset.eval_runtime \
  --prepared-countdown "$ARTIFACTS_DIR/countdown-test-hard.jsonl" \
  --model "$RLOO_DIR" \
  --output-dir "$EVAL_FINAL_DIR" \
  --max-new-tokens 384 \
  --max-clean-tries "$MAX_CLEAN_TRIES"
```

Optional smoke test: append `--max-examples 128` (or any N).

Optional (not paper baseline): append `--verifier-feedback` if you are experimenting with verifier hints on retries.

---

## Step 9 — Extension comparison

```bash
python3 scripts/run_extension_comparison.py \
  --prepared-countdown "$ARTIFACTS_DIR/countdown-test-hard.jsonl" \
  --results-jsonl "$EVAL_FINAL_DIR/results.jsonl" \
  --output-dir "$EXTENSIONS_DIR" \
  --max-cleans "$MAX_CLEAN_TRIES"
```

---

## Step 10 — Qualitative export

```bash
python3 scripts/export_qualitative_samples.py \
  --eval-results "$EVAL_FINAL_DIR/results.jsonl" \
  --output-path "$QUALITATIVE_PATH"
```

---

## Outputs to check

- **Summary metrics:** `$EVAL_FINAL_DIR/summary.json`
- **Per-row results:** `$EVAL_FINAL_DIR/results.jsonl`
- **Qualitative:** `$QUALITATIVE_PATH`

---

## Equivalence with the orchestrator

The same budget is expressible as env vars when calling the bundled script (see `scripts/replicate_paper.sh` header):

```bash
export LTR_SFT_MAX_TRAIN_EXAMPLES=8000
export LTR_EVAL_RAW_MAX_EXAMPLES=500
export LTR_MERGE_MAX_BASE_EXAMPLES=7500
export LTR_MERGE_MAX_MINED_EXAMPLES=500
export LTR_RLOO_MAX_TRAIN_EXAMPLES=8000
export LTR_RLOO_MAX_VALIDATION_EXAMPLES=25000
# Step 8: leave LTR_EVAL_FINAL_MAX_EXAMPLES unset for full hard eval

bash scripts/replicate_paper.sh \
  --base-model "$BASE_MODEL" \
  --scale-override "$SCALE" \
  --out-dir "$OUT_DIR" \
  --rloo-steps "$RLOO_STEPS"
```

---

## Troubleshooting

- **`ModuleNotFoundError: learning_to_reset`** — Run `pip install -e .` from the repo root with the active interpreter.
- **Mining / contamination errors** — Step 4 must use **`countdown-train.jsonl`** (prepared), not the hard test file.
- **Disk / cache** — Paper-scale runs need large HF cache and checkpoint space.

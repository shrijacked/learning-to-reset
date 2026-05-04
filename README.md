# Learning to Reset

[![CI](https://github.com/shrijacked/learning-to-reset/actions/workflows/ci.yml/badge.svg)](https://github.com/shrijacked/learning-to-reset/actions/workflows/ci.yml)

Research code for **reset-aware mathematical reasoning**: models learn to emit a `<clean>` token to clear cluttered scratch work and retry from a fresh prompt, with supervised fine-tuning (SFT), mined recovery traces, **modified RLOO**, and Countdown-style evaluation (including a hard slice).

---

## Get the code and run everything

These steps assume a Unix-like shell (Linux or macOS). **Windows:** use WSL2.

### 1. Clone

```bash
git clone https://github.com/shrijacked/learning-to-reset.git
cd learning-to-reset
```

### 2. Run the automated pipeline (recommended)

The repository entrypoint installs dependencies in a virtual environment, runs the **full** replication script (`scripts/replicate_paper.sh`: fetch data → prepare artifacts → SFT → eval → mine → re-SFT → RLOO → final eval → extensions), builds figures, and writes artifacts under `outputs/`.

```bash
chmod +x run.sh scripts/bootstrap_ubuntu.sh
bash run.sh
```

- **First-time / grading containers:** the default is a **small pilot** run (`Qwen/Qwen2.5-0.5B`, CPU-friendly caps). This needs **internet access** to download models and datasets from Hugging Face.
- **Logs:** `outputs/logs/run.log`
- **Main results:** `outputs/replicate-paper/eval-final/summary.json`, `outputs/replicate-paper/eval-final/results.jsonl`, `outputs/replicate-paper/qualitative.md`
- **Figures:** `outputs/figures/` and `outputs/figures/budget/`
- **Index of paths:** `outputs/ARTIFACTS.txt`

On **Debian/Ubuntu**, if `python3 -m venv` fails, install the venv package, then re-run:

```bash
sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
```

### 3. Optional: full-scale replication (long-running, needs a GPU)

For the paper-style data scale and a larger base model (heavy download and training):

```bash
RUN_FULL=1 bash run.sh
```

Set `HF_TOKEN` if your Hugging Face account requires it for gated assets:

```bash
export HF_TOKEN="hf_..."
RUN_FULL=1 bash run.sh
```

See **[docs/paper-replication.md](docs/paper-replication.md)** for wall-clock expectations and hardware notes.

### Environment variables (all optional)

| Variable | Effect |
|----------|--------|
| `RUN_FULL=1` | Paper data scale + larger default model (`run.sh`); long runtime. |
| `RUN_TESTS=1` | Run the unit test suite before the pipeline. |
| `SKIP_NETWORK_CHECK=1` | Skip the Hugging Face connectivity check at startup. |
| `HF_TOKEN` | Sent as `HUGGING_FACE_HUB_TOKEN` for Hugging Face Hub. |
| `BASE_MODEL` | Override the Hugging Face model id (defaults depend on `RUN_FULL`). |
| `LTR_RLOO_STEPS`, `LTR_EVAL_RAW_MAX_EXAMPLES`, `LTR_SFT_MAX_TRAIN_EXAMPLES`, `LTR_MERGE_MAX_*`, `LTR_RLOO_MAX_*`, `LTR_EVAL_FINAL_MAX_EXAMPLES`, `LTR_FORCE_CPU`, … | Passed through to `scripts/replicate_paper.sh`; see the header comment in that file. |

---

## Manual setup (developers)

If you prefer not to use `run.sh`:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
export PYTHONPATH=src
```

Run tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -q
```

Run the small built-in demo:

```bash
PYTHONPATH=src python -m learning_to_reset
```

Regenerate figures from bundled result tables (writes `docs/figures/` and `docs/figures/budget/`):

```bash
PYTHONPATH=src python scripts/build_figures.py --output-dir docs/figures --budget-dir docs/figures/budget
```

Drive only the orchestrator (you must install deps yourself):

```bash
PYTHONPATH=src bash scripts/replicate_paper.sh \
  --base-model Qwen/Qwen2.5-0.5B \
  --scale-override pilot \
  --out-dir runs/my-run \
  --rloo-steps 15
```

CLI-only validation without downloading models:

```bash
PYTHONPATH=src bash scripts/replicate_paper.sh --dry-run \
  --base-model Qwen/Qwen2.5-0.5B \
  --out-dir /tmp/ltr-dryrun
```

---

## Results and write-ups

- **[pre-readme.md](pre-readme.md)** — Formal summary tables and interpretation for the constrained replication track (figures match `docs/figures/budget/` when regenerated).
- **[result.md](result.md)** — Scratch log of console metrics from local runs (optional).


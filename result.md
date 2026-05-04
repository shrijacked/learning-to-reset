# Reconstruction of results from chat (lost run)

All values below were **pasted or inferred in conversation**, not read from your machine. Paths use your run root: `/root/runs/replicate-paper-2026-05-04-1520` (you also used `$OUT` for the same).

---

## Merge SFT corpus (`merge_sft_corpus`)

**Full merge (before capping):**

```text
[merge_sft_corpus] wrote 51629 examples (51129 from --sft-train, 500 from mined traces)
-> .../sft-train-combined.jsonl
```

**Capped merge (7500 base + 500 mined):**

```text
[merge_sft_corpus] wrote 8000 examples (7500 from --sft-train, 500 from mined traces)
-> /root/runs/replicate-paper-2026-05-04-1520/sft-train-combined.jsonl
```

---

## Step 6 — Second SFT (`sft_runtime` on combined 8k)

Trainer-style logs:

```text
{'loss': '1.317', 'grad_norm': '16.5', 'learning_rate': '2.5e-09', 'epoch': '1'}
{'eval_loss': '0.9366', 'eval_runtime': '72.07', 'eval_samples_per_second': '37.34', 'eval_steps_per_second': '37.34', 'epoch': '1'}
{'train_runtime': '926.6', 'train_samples_per_second': '8.634', 'train_steps_per_second': '8.634', 'train_loss': '0.8559', 'epoch': '1'}
100%|...| 8000/8000 [15:26<00:00,  8.63it/s]
Writing model shards: 100%|...| 1/1 [00:03<00:00,  3.37s/it]
```

Summary JSON printed at end:

```json
{
  "train_examples": 8000,
  "validation_examples": 2691,
  "train_loss": 0.8559386943413693,
  "output_dir": "/root/runs/replicate-paper-2026-05-04-1520/sft-mined"
}
```

**Model:** `Qwen/Qwen2.5-1.5B-Instruct` (per your commands).

---

## Step 7 — RLOO (`rloo_runtime`)

**Warnings (benign, repeated in thread):**

```text
RuntimeWarning: 'learning_to_reset.rloo_runtime' found in sys.modules after import of package 'learning_to_reset', but prior to execution of 'learning_to_reset.rloo_runtime'; this may result in unpredictable behaviour
```

**One observed startup log (200 steps, 8k train cap, full val load):**

```text
Loading weights: 100%|...| 338/338 [00:00<00:00, 6722.12it/s]
[rloo_runtime] model ready on cuda (/root/runs/replicate-paper-2026-05-04-1520/sft-mined); loading datasets next
[rloo_runtime] loading training examples from .../artifacts/countdown-train.jsonl, first 8000 rows only
[rloo_runtime] loading validation examples from .../artifacts/countdown-validation.jsonl
[rloo_runtime] loaded 8000 train, 25000 validation examples; starting 200 step(s) on cuda
[rloo_runtime] step 1/200
...
```

**Observed pacing (your note):** on the order of **~30 seconds per optimization step** with `--responses-per-prompt 4` and `--max-new-tokens 384`.

**Sample lines from `metrics.jsonl` (steps 1–5; you pasted step 1 explicitly and noted 2–5 were analogous):**

```json
{"step": 1, "loss": 0.0, "accuracy": 0.0, "valid_rate": 0.0, "average_score": 0.0, "clean_rate": 1.0, "average_initial_tokens": 42.5, "average_retry_tokens": 128.0, "average_total_tokens": 170.5, "average_advantage": 0.0, "reward_mode": "correctness"}
{"step": 2, "loss": 0.0, "accuracy": 0.0, "valid_rate": 0.0, "average_score": 0.0, "clean_rate": 1.0, "average_initial_tokens": 42.0, "average_retry_tokens": 128.0, "average_total_tokens": 170.0, "average_advantage": 0.0, "reward_mode": "correctness"}
{"step": 3, "loss": 0.0, "accuracy": 0.0, "valid_rate": 0.0, "average_score": 0.0, "clean_rate": 1.0, "average_initial_tokens": 42.0, "average_retry_tokens": 128.0, "average_total_tokens": 170.0, "average_advantage": 0.0, "reward_mode": "correctness"}
{"step": 4, "loss": 0.0, "accuracy": 0.0, "valid_rate": 0.0, "average_score": 0.0, "clean_rate": 1.0, "average_initial_tokens": 42.0, "average_retry_tokens": 128.0, "average_total_tokens": 170.0, "average_advantage": 0.0, "reward_mode": "correctness"}
{"step": 5, "loss": 0.0, "accuracy": 0.0, "valid_rate": 0.0, "average_score": 0.0, "clean_rate": 1.0, "average_initial_tokens": 42.0, "average_retry_tokens": 128.0, "average_total_tokens": 170.0, "average_advantage": 0.0, "reward_mode": "correctness"}
```

**Config you discussed later (not necessarily what produced the logs above):** e.g. 75 or 100 steps, `--responses-per-prompt` 2 vs 4, `--max-validation-examples` to shrink val RAM. Final RLOO `training-summary.json` / completed `metrics.jsonl` were **not** pasted in chat.

---

## Step 8 — Final eval (`eval_runtime`)

**Not in chat:** no `summary.json`, accuracy, or `results.jsonl` excerpts were shared.

**Command shape you were given:**

- `--prepared-countdown` → `$OUT/artifacts/countdown-test-hard.jsonl`
- `--model` → `$OUT/rloo`
- `--output-dir` → `$OUT/eval-final`
- `--max-new-tokens 384`, `--max-clean-tries 3`

---

## Steps 9–10

Extension comparison and qualitative export: **no outputs** shared in chat.

---

## How to recover on disk (if the run directory still exists)

If `/root/runs/replicate-paper-2026-05-04-1520` (or your `$OUT`) is still available somewhere, prefer re-copying from:

| Step | Likely paths |
|------|----------------|
| Merge | `sft-train-combined.jsonl` |
| SFT | `sft-mined/training-summary.json`, trainer state under `sft-mined/` |
| RLOO | `rloo/metrics.jsonl`, `rloo/training-summary.json`, `rloo/final-checkpoint/` |
| Eval | `eval-final/summary.json`, `eval-final/results.jsonl` |
| Artifacts | `artifacts/manifest.json` |

This file is only a **chat backup**; treat numbers as approximate if you edited commands between messages.

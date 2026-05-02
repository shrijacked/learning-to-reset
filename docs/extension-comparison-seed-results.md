# Extension Comparison Seed Results

Generated during Phase 1 completion.

## Seed 131

Command:

```bash
PYTHONPATH=src python3 scripts/run_extension_comparison.py \
  --prepared-countdown tmp/paper-artifacts-grounded-26apr-seed131/countdown-test-hard.jsonl \
  --results-jsonl tmp/paper-eval/sft-grounded-26apr-seed219mine-seed131-multiclean3/results.jsonl \
  --output-dir tmp/paper-eval/extensions-seed131
```

Result:

| Metric | Value |
|---|---:|
| total examples | 32 |
| skipped incomplete clean segments | 0 |
| full reset wins | 32 |
| full reset mean adjusted reward | 0.0500 |
| selective retention mean adjusted reward | 0.0500 |
| memory mean adjusted reward | 0.0500 |

Interpretation: all controllers tie because the underlying generated segments remain
validity-heavy but target-correctness-poor.

## Seed 219

Command:

```bash
PYTHONPATH=src python3 scripts/run_extension_comparison.py \
  --prepared-countdown tmp/paper-artifacts-grounded-26apr-seed219/countdown-test-hard.jsonl \
  --results-jsonl tmp/paper-eval/sft-grounded-26apr-seed219-raw/results.jsonl \
  --output-dir tmp/paper-eval/extensions-seed219
```

Result:

| Metric | Value |
|---|---:|
| total comparable examples | 0 |
| skipped incomplete clean segments | 32 |

Interpretation: the local seed-219 artifact is a raw first-segment eval. Every
example requests `<clean>` without carrying follow-up retry segments, so it is not a
valid controller-comparison input. The runner now reports that case instead of
crashing.

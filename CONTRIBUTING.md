# Contributing

This project is a research replication workbench. Keep changes small, tested, and
honest about what they prove.

## Local Checks

```bash
make test
make figures
LTR_REPLICATE_DRY_RUN=1 PYTHONPATH=src python3 -m unittest tests.test_replicate_paper_dry_run
```

## Rules Of Thumb

- Add a regression test before fixing a bug.
- Do not report 0.5B local results as paper-faithful 1B replication results.
- Keep mined recovery examples disjoint from the holdout used for reporting.
- Preserve provenance in `manifest.json` and `sft-mix-summary.json` when changing data prep.
- Use `scripts/replicate_paper.sh --dry-run` after changing CLI wiring.

## Reporting Results

Include the exact command, hardware, model path, summary JSON, and any deviation
from `docs/paper-replication.md`. If a run improves validity but not correctness,
say that plainly.

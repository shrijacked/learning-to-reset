#!/usr/bin/env python
"""CLI wrapper around ``learning_to_reset.extension_comparison_runner``.

Reads a multi-clean eval's ``results.jsonl`` and the matching prepared
Countdown JSONL, then scores the same response sequence under three
controllers:

* ``full_reset`` — the baseline ``manage_bounded_clean_cycles`` policy.
* ``selective_retention`` — keeps explicit retained notes across resets.
* ``memory`` — recalls top-N salient memory entries written so far.

This is a *controller comparison* using identical generations, not a fully
isolated A/B test that re-rolls the model per strategy. The aim is to expose
the strategy-only delta on the existing fresh-32 holdout outputs without
spending more GPU/MPS time.

Outputs:

* ``<output-dir>/extension-comparison.json`` — flat per-strategy summary plus
  every example's row breakdown.
* ``<output-dir>/extension-comparison.md`` — human-readable Figure-6-style
  table.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if REPO_SRC.exists() and str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from learning_to_reset.extension_comparison_runner import main


if __name__ == "__main__":
    raise SystemExit(main())

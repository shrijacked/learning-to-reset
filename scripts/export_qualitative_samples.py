#!/usr/bin/env python3
"""Thin wrapper around `python -m learning_to_reset.qualitative_export`.

Useful when the user prefers a discoverable script path. All real logic
lives in the qualitative_export module so it can be unit-tested without
running through subprocess.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))

    from learning_to_reset.qualitative_export import main as module_main

    return module_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

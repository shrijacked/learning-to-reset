"""Gated subprocess test for scripts/replicate_paper.sh --dry-run.

The full pipeline cannot run inside the test suite (model downloads,
hours of GPU compute). What we *can* verify on every commit is that
the orchestrator script still parses every wired CLI. This test is
opt-in via the ``LTR_REPLICATE_DRY_RUN=1`` environment variable so the
default ``python -m unittest discover`` stays fast and offline.

Run it with::

    LTR_REPLICATE_DRY_RUN=1 PYTHONPATH=src \\
        .venv/bin/python3 -m unittest tests.test_replicate_paper_dry_run

The replication runbook (``docs/paper-replication.md``) documents the
same command for downstream users.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "replicate_paper.sh"


def _bash_executable() -> str:
    """Return the bash executable to use, preferring /bin/bash on macOS."""

    candidate = shutil.which("bash")
    if not candidate:
        raise unittest.SkipTest("bash not available on PATH; cannot run shell script.")
    return candidate


@unittest.skipUnless(
    os.environ.get("LTR_REPLICATE_DRY_RUN") == "1",
    "set LTR_REPLICATE_DRY_RUN=1 to opt into the replicate_paper.sh dry-run check.",
)
class ReplicatePaperDryRunTests(unittest.TestCase):
    """Subprocess test that asserts every CLI invoked by the script parses.

    These tests do not download models or datasets; they only execute the
    bash orchestrator with --dry-run, which calls each command's --help.
    """

    def test_dry_run_exits_zero_with_qwen_0_5b_base(self) -> None:
        self.assertTrue(SCRIPT_PATH.exists(), f"script missing: {SCRIPT_PATH}")
        bash = _bash_executable()
        with tempfile.TemporaryDirectory(prefix="ltr-replicate-dryrun-") as tmpdir:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT / "src")
            env.setdefault("PYTHON", sys.executable)
            result = subprocess.run(
                [
                    bash,
                    str(SCRIPT_PATH),
                    "--dry-run",
                    "--base-model",
                    "Qwen/Qwen2.5-0.5B",
                    "--out-dir",
                    str(Path(tmpdir) / "replicate-out"),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
                cwd=REPO_ROOT,
            )
        self.assertEqual(
            result.returncode,
            0,
            msg=(
                "replicate_paper.sh --dry-run failed.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            ),
        )
        self.assertIn("dry-run complete.", result.stdout)
        self.assertNotIn("ERROR:", result.stdout + result.stderr)

    def test_dry_run_reports_each_pipeline_stage(self) -> None:
        self.assertTrue(SCRIPT_PATH.exists())
        bash = _bash_executable()
        with tempfile.TemporaryDirectory(prefix="ltr-replicate-dryrun-stages-") as tmpdir:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT / "src")
            env.setdefault("PYTHON", sys.executable)
            result = subprocess.run(
                [
                    bash,
                    str(SCRIPT_PATH),
                    "--dry-run",
                    "--base-model",
                    "Qwen/Qwen2.5-0.5B",
                    "--out-dir",
                    str(Path(tmpdir) / "replicate-out"),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
                cwd=REPO_ROOT,
            )
        for stage in (
            "paper_sources",
            "prepare_paper_artifacts",
            "sft_runtime",
            "eval_runtime",
            "failure_recovery_traces",
            "rloo_runtime",
        ):
            self.assertIn(
                f"cli ok: {stage}",
                result.stdout,
                msg=f"dry-run did not validate the {stage} CLI",
            )

    def test_dry_run_exits_zero_with_ltr_verifier_feedback_env(self) -> None:
        """Orchestrator must stay valid when step 8 may add --verifier-feedback."""

        self.assertTrue(SCRIPT_PATH.exists())
        bash = _bash_executable()
        with tempfile.TemporaryDirectory(prefix="ltr-replicate-dryrun-vf-") as tmpdir:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT / "src")
            env.setdefault("PYTHON", sys.executable)
            env["LTR_VERIFIER_FEEDBACK"] = "1"
            result = subprocess.run(
                [
                    bash,
                    str(SCRIPT_PATH),
                    "--dry-run",
                    "--base-model",
                    "Qwen/Qwen2.5-0.5B",
                    "--out-dir",
                    str(Path(tmpdir) / "replicate-out"),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
                cwd=REPO_ROOT,
            )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("dry-run complete.", result.stdout)


if __name__ == "__main__":
    unittest.main()

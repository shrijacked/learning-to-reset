import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from learning_to_reset.compare_eval_results import (
    build_evaluation_comparison,
    load_evaluation_summary,
    main,
    render_comparison_markdown,
    write_comparison_outputs,
)


class CompareEvalResultsTests(unittest.TestCase):
    def test_build_evaluation_comparison_computes_candidate_deltas(self) -> None:
        comparison = build_evaluation_comparison(
            baseline_summary={
                "total_examples": 8,
                "correct": 0,
                "valid": 0,
                "accuracy": 0.0,
                "valid_rate": 0.0,
                "average_score": 0.0,
            },
            candidate_summary={
                "total_examples": 8,
                "correct": 1,
                "valid": 8,
                "accuracy": 0.125,
                "valid_rate": 1.0,
                "average_score": 0.225,
                "clean_rate": 1.0,
            },
            baseline_label="raw",
            candidate_label="reset-aware",
        )

        self.assertEqual(comparison["winner"], "reset-aware")
        self.assertEqual(comparison["delta"]["accuracy"], 0.125)
        self.assertEqual(comparison["delta"]["valid_rate"], 1.0)
        self.assertIsNone(comparison["delta"]["clean_rate"])

    def test_build_evaluation_comparison_flags_mismatched_totals(self) -> None:
        comparison = build_evaluation_comparison(
            baseline_summary={"total_examples": 4, "accuracy": 0.25},
            candidate_summary={"total_examples": 8, "accuracy": 0.25},
        )

        self.assertEqual(comparison["winner"], "tie")
        self.assertEqual(
            comparison["notes"],
            ["Evaluation totals differ; compare rates rather than raw counts."],
        )

    def test_write_comparison_outputs_creates_json_and_markdown(self) -> None:
        comparison = build_evaluation_comparison(
            baseline_summary={"total_examples": 8, "accuracy": 0.0},
            candidate_summary={"total_examples": 8, "accuracy": 0.125},
        )

        with tempfile.TemporaryDirectory() as tmp:
            write_comparison_outputs(comparison, tmp)
            comparison_json = Path(tmp) / "comparison.json"
            comparison_md = Path(tmp) / "comparison.md"

            self.assertTrue(comparison_json.exists())
            self.assertTrue(comparison_md.exists())
            self.assertEqual(json.loads(comparison_json.read_text())["winner"], "candidate")
            self.assertIn("| `accuracy` | 0.0 | 0.125 | 0.125 |", comparison_md.read_text())

    def test_main_loads_eval_dirs_and_writes_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_dir = root / "raw"
            candidate_dir = root / "reset"
            output_dir = root / "comparison"
            baseline_dir.mkdir()
            candidate_dir.mkdir()
            (baseline_dir / "summary.json").write_text(
                json.dumps({"total_examples": 8, "accuracy": 0.0}),
                encoding="utf-8",
            )
            (candidate_dir / "summary.json").write_text(
                json.dumps({"total_examples": 8, "accuracy": 0.125}),
                encoding="utf-8",
            )

            with redirect_stdout(StringIO()):
                exit_code = main(
                    (
                        "--baseline-dir",
                        str(baseline_dir),
                        "--candidate-dir",
                        str(candidate_dir),
                        "--output-dir",
                        str(output_dir),
                        "--baseline-label",
                        "raw",
                        "--candidate-label",
                        "reset-aware",
                    )
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                load_evaluation_summary(baseline_dir),
                {"total_examples": 8, "accuracy": 0.0},
            )
            self.assertIn("reset-aware", (output_dir / "comparison.md").read_text())

    def test_render_comparison_markdown_includes_notes(self) -> None:
        comparison = build_evaluation_comparison(
            baseline_summary={"total_examples": 4, "accuracy": 0.0},
            candidate_summary={"total_examples": 8, "accuracy": 0.0},
        )

        markdown = render_comparison_markdown(comparison)

        self.assertIn("## Notes", markdown)
        self.assertIn("Evaluation totals differ", markdown)


if __name__ == "__main__":
    unittest.main()

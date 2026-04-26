"""Red-phase tests for the extension comparison runner.

The B3 wrapper reads a multi-clean eval's ``results.jsonl`` (which already
contains per-example segment responses) and feeds the same response sequence
to all three controllers in
``learning_to_reset.extension_comparison.compare_extension_trajectories``.
It then aggregates winners and average adjusted reward per strategy.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from learning_to_reset.extension_comparison_runner import (
    aggregate_extension_comparisons,
    iter_examples_with_segments,
    run_extension_comparison_on_files,
)


def _write_jsonl(path: Path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


class IterExamplesWithSegmentsTests(unittest.TestCase):
    def test_pairs_prepared_examples_with_eval_segments_by_source_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prepared = tmp_path / "countdown.jsonl"
            results = tmp_path / "results.jsonl"
            _write_jsonl(
                prepared,
                [
                    {
                        "prompt": "p1",
                        "response": "",
                        "metadata": {
                            "source_id": "ex-1",
                            "numbers": [1, 2, 3, 4],
                            "target": 24,
                        },
                    },
                    {
                        "prompt": "p2",
                        "response": "",
                        "metadata": {
                            "source_id": "ex-2",
                            "numbers": [5, 6, 7, 8],
                            "target": 30,
                        },
                    },
                ],
            )
            _write_jsonl(
                results,
                [
                    {
                        "source_id": "ex-1",
                        "segments": [
                            {"response": "<think> 1*2*3*4=24 </think><answer>1*2*3*4</answer>"}
                        ],
                    },
                    {
                        "source_id": "ex-2",
                        "segments": [
                            {"response": "<think> 5+6=11 </think><clean>"},
                            {"response": "<think> retry </think><answer>5+6+7+12</answer>"},
                        ],
                    },
                ],
            )
            paired = list(
                iter_examples_with_segments(
                    prepared_path=str(prepared),
                    results_path=str(results),
                )
            )
        self.assertEqual(len(paired), 2)
        self.assertEqual(paired[0].sample.source_id, "ex-1")
        self.assertEqual(len(paired[0].responses), 1)
        self.assertEqual(len(paired[1].responses), 2)


class AggregateExtensionComparisonsTests(unittest.TestCase):
    def test_summary_counts_winners_and_averages_reward(self) -> None:
        from learning_to_reset.extension_comparison import (
            ExtensionComparison,
            ExtensionComparisonRow,
        )

        comparisons = [
            ExtensionComparison(
                rows=(
                    ExtensionComparisonRow(
                        mode="full_reset",
                        clean_count=1,
                        budget_exhausted=False,
                        final_answer="1+2",
                        total_reward=1.0,
                        adjusted_total_reward=1.0,
                    ),
                    ExtensionComparisonRow(
                        mode="selective_retention",
                        clean_count=1,
                        budget_exhausted=False,
                        final_answer="1+2",
                        total_reward=0.8,
                        adjusted_total_reward=0.8,
                    ),
                    ExtensionComparisonRow(
                        mode="memory",
                        clean_count=1,
                        budget_exhausted=False,
                        final_answer="1+2",
                        total_reward=0.5,
                        adjusted_total_reward=0.5,
                    ),
                ),
                winner="full_reset",
            ),
            ExtensionComparison(
                rows=(
                    ExtensionComparisonRow(
                        mode="full_reset",
                        clean_count=2,
                        budget_exhausted=True,
                        final_answer=None,
                        total_reward=0.0,
                        adjusted_total_reward=0.0,
                    ),
                    ExtensionComparisonRow(
                        mode="selective_retention",
                        clean_count=2,
                        budget_exhausted=True,
                        final_answer=None,
                        total_reward=0.0,
                        adjusted_total_reward=0.0,
                    ),
                    ExtensionComparisonRow(
                        mode="memory",
                        clean_count=2,
                        budget_exhausted=False,
                        final_answer="x*y",
                        total_reward=1.0,
                        adjusted_total_reward=0.9,
                    ),
                ),
                winner="memory",
            ),
        ]
        summary = aggregate_extension_comparisons(comparisons)
        self.assertEqual(summary["total_examples"], 2)
        self.assertEqual(summary["winners"]["full_reset"], 1)
        self.assertEqual(summary["winners"]["memory"], 1)
        self.assertNotIn("selective_retention", summary["winners"])
        self.assertAlmostEqual(
            summary["mean_adjusted_reward"]["full_reset"], 0.5
        )
        self.assertAlmostEqual(
            summary["mean_adjusted_reward"]["memory"], 0.7
        )


class RunOnFilesIntegrationTests(unittest.TestCase):
    def test_run_extension_comparison_on_files_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prepared = tmp_path / "countdown.jsonl"
            results = tmp_path / "results.jsonl"
            output_dir = tmp_path / "out"
            _write_jsonl(
                prepared,
                [
                    {
                        "prompt": "Use 1, 2, 3, 4 to reach 24.",
                        "response": "",
                        "metadata": {
                            "source_id": "ex-1",
                            "numbers": [1, 2, 3, 4],
                            "target": 24,
                            "question": "Use 1, 2, 3, 4 to reach 24.",
                        },
                    }
                ],
            )
            _write_jsonl(
                results,
                [
                    {
                        "source_id": "ex-1",
                        "segments": [
                            {
                                "response": (
                                    "<think>1*2*3*4=24</think>"
                                    "<answer>1*2*3*4</answer>"
                                )
                            }
                        ],
                    }
                ],
            )
            summary = run_extension_comparison_on_files(
                prepared_path=str(prepared),
                results_path=str(results),
                output_dir=str(output_dir),
                max_cleans=2,
            )
            self.assertEqual(summary["total_examples"], 1)
            self.assertTrue((output_dir / "extension-comparison.json").exists())
            self.assertTrue((output_dir / "extension-comparison.md").exists())


if __name__ == "__main__":
    unittest.main()

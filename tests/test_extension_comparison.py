import tempfile
import unittest
from pathlib import Path

from learning_to_reset.data import CountdownSample
from learning_to_reset.extension_comparison import (
    compare_extension_trajectories,
    render_extension_comparison_markdown,
    write_extension_comparison_outputs,
)


class ExtensionComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = CountdownSample(
            source_id="countdown-1",
            numbers=(60, 27, 19),
            target=68,
            question="Reach 68 using 60, 27, 19.",
        )

    def test_compare_extension_trajectories_selects_highest_adjusted_reward(self) -> None:
        comparison = compare_extension_trajectories(
            sample=self.sample,
            base_instructions="Solve carefully.",
            clean_instructions="Emit <clean> if needed.",
            max_cleans=1,
            full_reset_responses=(
                "<think>Messy.</think><clean>",
                "<think>Wrong.</think><answer>60 + 27</answer>",
            ),
            selective_retention_responses=(
                "<think>Keep useful sum.</think><retain>60 + 27 = 87.</retain><clean>",
                "<think>87 - 19 = 68.</think><answer>(60 + 27) - 19</answer>",
            ),
            memory_responses=(
                '<memory tags="countdown">60 + 27 = 87.</memory><clean>',
                "<think>87 - 19 = 68.</think><answer>(60 + 27) - 19</answer>",
            ),
        )

        self.assertEqual(comparison.winner, "selective_retention")
        self.assertEqual(len(comparison.rows), 3)
        self.assertAlmostEqual(comparison.rows[0].adjusted_total_reward, 0.05)
        self.assertAlmostEqual(comparison.rows[1].adjusted_total_reward, 1.25)
        self.assertAlmostEqual(comparison.rows[2].adjusted_total_reward, 1.25)

    def test_render_extension_comparison_markdown_includes_mode_table(self) -> None:
        comparison = compare_extension_trajectories(
            sample=self.sample,
            base_instructions="Solve carefully.",
            clean_instructions="Emit <clean> if needed.",
            max_cleans=1,
            full_reset_responses=("<think>Wrong.</think><answer>60 + 27</answer>",),
            selective_retention_responses=(
                "<think>Correct.</think><answer>(60 + 27) - 19</answer>",
            ),
            memory_responses=(
                "<think>Correct.</think><answer>(60 + 27) - 19</answer>",
            ),
        )

        markdown = render_extension_comparison_markdown(comparison)

        self.assertIn("| Mode | Clean count | Final answer | Adjusted reward |", markdown)
        self.assertIn("selective_retention", markdown)
        self.assertIn("memory", markdown)

    def test_write_extension_comparison_outputs_writes_json_and_markdown(self) -> None:
        comparison = compare_extension_trajectories(
            sample=self.sample,
            base_instructions="Solve carefully.",
            clean_instructions="Emit <clean> if needed.",
            max_cleans=1,
            full_reset_responses=("<think>Wrong.</think><answer>60 + 27</answer>",),
            selective_retention_responses=(
                "<think>Correct.</think><answer>(60 + 27) - 19</answer>",
            ),
            memory_responses=(
                "<think>Correct.</think><answer>(60 + 27) - 19</answer>",
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_paths = write_extension_comparison_outputs(
                comparison,
                output_dir=Path(tmp_dir),
            )
            json_text = output_paths["json"].read_text(encoding="utf-8")
            markdown_text = output_paths["markdown"].read_text(encoding="utf-8")

        self.assertIn('"winner": "selective_retention"', json_text)
        self.assertIn("# Extension Comparison", markdown_text)


if __name__ == "__main__":
    unittest.main()

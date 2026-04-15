import unittest

from learning_to_reset.data import CountdownSample
from learning_to_reset.multi_clean_extension import (
    build_multi_clean_trajectory,
    manage_bounded_clean_cycles,
)


class MultiCleanExtensionTests(unittest.TestCase):
    def test_manage_bounded_clean_cycles_allows_multiple_retries_within_budget(self) -> None:
        managed = manage_bounded_clean_cycles(
            question="Reach 68 using 60, 27, 19.",
            base_instructions="Solve carefully.",
            clean_instructions="Emit <clean> if needed.",
            responses=(
                "<think>First search is messy.</think><clean>",
                "<think>Second search is still messy.</think><clean>",
                "<think>Fresh start.</think><answer>(60 + 27) - 19</answer>",
            ),
            max_cleans=2,
        )

        self.assertEqual(managed.clean_count, 2)
        self.assertFalse(managed.budget_exhausted)
        self.assertEqual(managed.final_answer, "(60 + 27) - 19")
        self.assertEqual(len(managed.segments), 3)

    def test_manage_bounded_clean_cycles_stops_when_budget_is_exhausted(self) -> None:
        managed = manage_bounded_clean_cycles(
            question="Reach 68 using 60, 27, 19.",
            base_instructions="Solve carefully.",
            clean_instructions="Emit <clean> if needed.",
            responses=(
                "<think>First search is messy.</think><clean>",
                "<think>Still messy.</think><clean>",
            ),
            max_cleans=1,
        )

        self.assertEqual(managed.clean_count, 1)
        self.assertTrue(managed.budget_exhausted)
        self.assertEqual(managed.final_response, "<think>Still messy.</think><clean>")

    def test_manage_bounded_clean_cycles_requires_follow_up_while_budget_remains(self) -> None:
        with self.assertRaises(ValueError):
            manage_bounded_clean_cycles(
                question="Reach 68 using 60, 27, 19.",
                base_instructions="Solve carefully.",
                clean_instructions="Emit <clean> if needed.",
                responses=("<think>Messy.</think><clean>",),
                max_cleans=1,
            )

    def test_build_multi_clean_trajectory_applies_clean_step_penalty(self) -> None:
        sample = CountdownSample(
            source_id="countdown-1",
            numbers=(60, 27, 19),
            target=68,
            question="Reach 68 using 60, 27, 19.",
        )

        trajectory = build_multi_clean_trajectory(
            question=sample.question,
            sample=sample,
            responses=(
                "<think>Messy search.</think><clean>",
                "<think>Fresh.</think><answer>(60 + 27) - 19</answer>",
            ),
            max_cleans=1,
            clean_step_penalty=0.05,
            base_instructions="Solve carefully.",
            clean_instructions="Emit <clean> if needed.",
        )

        self.assertEqual(trajectory.managed.clean_count, 1)
        self.assertAlmostEqual(trajectory.segment_rewards[-1].total_reward, 1.1)
        self.assertAlmostEqual(trajectory.clean_penalty, 0.05)
        self.assertAlmostEqual(trajectory.adjusted_total_reward, 1.05)


if __name__ == "__main__":
    unittest.main()

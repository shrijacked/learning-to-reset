import math
import unittest

from learning_to_reset.data import CountdownSample
from learning_to_reset.rollout_runtime import (
    build_clean_trajectory,
    compute_countdown_reward,
)


class RolloutRuntimeTests(unittest.TestCase):
    def test_compute_countdown_reward_combines_format_and_correctness(self) -> None:
        sample = CountdownSample(
            source_id="c1",
            numbers=(60, 27, 19),
            target=68,
            question="Reach 68 using 60, 27, 19.",
        )

        reward = compute_countdown_reward(
            "<think>Try.</think><answer>(60 + 27) - 19</answer>",
            sample,
        )

        self.assertAlmostEqual(reward.total_reward, 1.1)
        self.assertTrue(reward.has_valid_format)
        self.assertTrue(reward.is_correct)

    def test_compute_countdown_reward_handles_missing_answer_format(self) -> None:
        sample = CountdownSample(
            source_id="c2",
            numbers=(25, 7, 3, 2),
            target=50,
            question="Reach 50 using 25, 7, 3, 2.",
        )

        reward = compute_countdown_reward("<think>No answer.</think>", sample)

        self.assertAlmostEqual(reward.total_reward, 0.0)
        self.assertFalse(reward.has_valid_format)
        self.assertFalse(reward.is_correct)

    def test_build_clean_trajectory_prefers_retry_reward(self) -> None:
        sample = CountdownSample(
            source_id="c3",
            numbers=(61, 63, 57),
            target=55,
            question="Reach 55 using 61, 63, 57.",
        )

        trajectory = build_clean_trajectory(
            question=sample.question,
            sample=sample,
            initial_response="<think>Confusing path.</think><clean>",
            retry_response="<think>Fresh.</think><answer>55</answer>",
        )

        self.assertTrue(trajectory.cleaned)
        self.assertEqual(trajectory.final_response, "<think>Fresh.</think><answer>55</answer>")
        self.assertTrue(math.isclose(trajectory.trajectory_sample.retry_reward or 0.0, 0.1))
        self.assertTrue(math.isclose(trajectory.final_reward.total_reward, 0.1))

    def test_build_clean_trajectory_accepts_explicit_token_lengths(self) -> None:
        sample = CountdownSample(
            source_id="c4",
            numbers=(60, 27, 19),
            target=68,
            question="Reach 68 using 60, 27, 19.",
        )

        trajectory = build_clean_trajectory(
            question=sample.question,
            sample=sample,
            initial_response="<think>Try.</think><answer>(60 + 27) - 19</answer>",
            initial_length=7,
        )

        self.assertEqual(trajectory.trajectory_sample.initial_length, 7)


if __name__ == "__main__":
    unittest.main()

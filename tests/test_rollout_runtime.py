import math
import unittest

from learning_to_reset.data import CountdownSample
from learning_to_reset.rollout_runtime import (
    build_clean_trajectory,
    compute_countdown_reward,
    score_arithmetic_claims,
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

    def test_score_arithmetic_claims_rewards_correct_claims_and_penalizes_wrong_ones(self) -> None:
        reward, total, correct, incorrect = score_arithmetic_claims(
            "<think>Compute 7 + 2 = 9. Compute 9 + 1 = 10.</think>"
        )

        self.assertEqual(total, 2)
        self.assertEqual(correct, 2)
        self.assertEqual(incorrect, 0)
        self.assertAlmostEqual(reward, 0.2)

        reward2, total2, correct2, incorrect2 = score_arithmetic_claims(
            "<think>Compute 7 + 2 = 8. Compute 9 + 1 = 10.</think>"
        )

        self.assertEqual(total2, 2)
        self.assertEqual(correct2, 1)
        self.assertEqual(incorrect2, 1)
        self.assertAlmostEqual(reward2, 0.0)

    def test_compute_countdown_reward_includes_arithmetic_claim_reward(self) -> None:
        sample = CountdownSample(
            source_id="c1b",
            numbers=(7, 2, 1),
            target=10,
            question="Reach 10 using 7, 2, 1.",
        )

        reward = compute_countdown_reward(
            "<think>Compute 7 + 2 = 9. Compute 9 + 1 = 10.</think><answer>((7 + 2) + 1)</answer>",
            sample,
        )

        self.assertAlmostEqual(reward.correctness_reward, 1.0)
        self.assertAlmostEqual(reward.arithmetic_claim_reward, 0.2)
        self.assertAlmostEqual(reward.total_reward, 1.3)
        self.assertEqual(reward.arithmetic_claims_total, 2)
        self.assertEqual(reward.arithmetic_claims_correct, 2)
        self.assertEqual(reward.arithmetic_claims_incorrect, 0)

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

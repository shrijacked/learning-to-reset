import math
import unittest

from learning_to_reset.rloo import (
    TrajectorySample,
    compute_leave_one_out_advantages,
    compute_modified_rloo_terms,
    compute_total_reward,
)


class ModifiedRLOOTests(unittest.TestCase):
    def test_total_reward_prefers_retry_reward_when_clean_occurs(self) -> None:
        self.assertEqual(compute_total_reward(initial_reward=0.2, retry_reward=0.9), 0.9)
        self.assertEqual(compute_total_reward(initial_reward=0.2, retry_reward=None), 0.2)

    def test_leave_one_out_advantages_match_formula(self) -> None:
        advantages = compute_leave_one_out_advantages([0.2, 1.0, 0.0])

        self.assertEqual(len(advantages), 3)
        self.assertTrue(math.isclose(advantages[0], -0.3))
        self.assertTrue(math.isclose(advantages[1], 0.9))
        self.assertTrue(math.isclose(advantages[2], -0.6))

    def test_modified_rloo_terms_share_advantage_across_segments(self) -> None:
        result = compute_modified_rloo_terms(
            [
                TrajectorySample(initial_reward=0.2, initial_length=2),
                TrajectorySample(
                    initial_reward=0.1,
                    initial_length=4,
                    retry_reward=1.0,
                    retry_length=5,
                ),
                TrajectorySample(
                    initial_reward=0.4,
                    initial_length=3,
                    retry_reward=0.0,
                    retry_length=6,
                ),
            ]
        )

        self.assertEqual(result.normalization, 5)
        self.assertEqual(result.clean_trajectory_count, 2)

        second_term = result.terms[1]
        self.assertTrue(math.isclose(second_term.total_reward, 1.0))
        self.assertTrue(math.isclose(second_term.advantage, 0.9))
        self.assertTrue(math.isclose(second_term.initial_scale, 0.045))
        self.assertTrue(math.isclose(second_term.retry_scale, 0.036))


if __name__ == "__main__":
    unittest.main()

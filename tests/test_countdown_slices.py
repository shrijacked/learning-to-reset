import unittest

from learning_to_reset.countdown_slices import (
    can_reach_with_add_sub_only,
    filter_hard_countdown_samples,
    is_hard_countdown_sample,
)
from learning_to_reset.data import CountdownSample


class CountdownSliceTests(unittest.TestCase):
    def test_can_reach_with_add_sub_only_allows_subsets(self) -> None:
        self.assertTrue(can_reach_with_add_sub_only((20, 5, 99), 25))
        self.assertFalse(can_reach_with_add_sub_only((6, 7), 42))

    def test_is_hard_countdown_sample_requires_mul_or_div_when_not_metadata_marked(self) -> None:
        easy = CountdownSample(
            numbers=(20, 5, 99),
            target=25,
            question="Reach 25 using 20, 5, 99.",
            source_id="easy",
        )
        hard = CountdownSample(
            numbers=(6, 7),
            target=42,
            question="Reach 42 using 6, 7.",
            source_id="hard",
        )

        self.assertFalse(is_hard_countdown_sample(easy))
        self.assertTrue(is_hard_countdown_sample(hard))

    def test_filter_hard_countdown_samples_preserves_order(self) -> None:
        samples = (
            CountdownSample(numbers=(1, 2), target=3, question="Reach 3.", source_id="easy-1"),
            CountdownSample(numbers=(8, 4), target=2, question="Reach 2.", source_id="hard-1"),
            CountdownSample(numbers=(9, 1), target=10, question="Reach 10.", source_id="easy-2"),
        )

        filtered = filter_hard_countdown_samples(samples)

        self.assertEqual(tuple(sample.source_id for sample in filtered), ("hard-1",))


if __name__ == "__main__":
    unittest.main()

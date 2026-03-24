import math
import unittest

from learning_to_reset.demo import build_demo_report, build_demo_snapshot


class DemoTests(unittest.TestCase):
    def test_demo_snapshot_matches_expected_baseline_values(self) -> None:
        snapshot = build_demo_snapshot()

        self.assertEqual(snapshot.normalized_answer, "42")
        self.assertTrue(snapshot.curated_uses_clean)
        self.assertEqual(snapshot.final_answer, "55")
        self.assertEqual(snapshot.rloo_normalization, 5)
        self.assertTrue(math.isclose(snapshot.second_advantage, 0.9))

    def test_demo_report_covers_all_baseline_mechanics(self) -> None:
        report = build_demo_report()

        self.assertIn("Trace Preparation", report)
        self.assertIn("Reset Flow", report)
        self.assertIn("Reward Terms", report)
        self.assertIn("<clean>", report)
        self.assertIn("Final answer: 55", report)


if __name__ == "__main__":
    unittest.main()

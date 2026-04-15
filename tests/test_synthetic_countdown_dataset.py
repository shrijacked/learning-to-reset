import json
import tempfile
import unittest
from pathlib import Path

from learning_to_reset.countdown_verifier import verify_countdown_expression
from learning_to_reset.data import load_countdown_samples
from learning_to_reset.synthetic_countdown_dataset import (
    build_sampling_profile,
    generate_synthetic_countdown_dataset,
)
from learning_to_reset.countdown_solver import solve_countdown


class SyntheticCountdownDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_records = [
            {
                "source_id": "ref-1",
                "numbers": [9, 11, 12, 17],
                "target": 70,
                "question": "Use the numbers 9, 11, 12, 17 to reach 70.",
            },
            {
                "source_id": "ref-2",
                "numbers": [78, 59, 2],
                "target": 98,
                "question": "Use the numbers 78, 59, 2 to reach 98.",
            },
        ]

    def test_build_sampling_profile_reflects_reference_slice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            reference_path = Path(tmp_dir) / "reference.jsonl"
            reference_path.write_text(
                "\n".join(json.dumps(record) for record in self.reference_records) + "\n",
                encoding="utf-8",
            )

            profile = build_sampling_profile(load_countdown_samples(reference_path))

        self.assertEqual(profile.arities, (4, 3))
        self.assertEqual(profile.min_target, 70)
        self.assertEqual(profile.max_target, 98)
        self.assertIn(78, profile.number_pool)
        self.assertIn(17, profile.number_pool)

    def test_generate_synthetic_countdown_dataset_writes_solved_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            reference_path = Path(tmp_dir) / "reference.jsonl"
            output_path = Path(tmp_dir) / "generated.jsonl"
            summary_path = output_path.with_suffix(".summary.json")
            reference_path.write_text(
                "\n".join(json.dumps(record) for record in self.reference_records) + "\n",
                encoding="utf-8",
            )

            summary = generate_synthetic_countdown_dataset(
                reference_path=reference_path,
                output_path=output_path,
                num_samples=4,
                seed=7,
            )
            generated_samples = load_countdown_samples(output_path)
            summary_exists = summary_path.exists()

        self.assertEqual(summary["samples_written"], 4)
        self.assertEqual(len(generated_samples), 4)
        self.assertTrue(summary_exists)

        arities = {len(sample.numbers) for sample in generated_samples}
        self.assertTrue(arities.issubset({3, 4}))

        for sample in generated_samples:
            expression = solve_countdown(sample.numbers, sample.target)
            self.assertIsNotNone(expression)
            verification = verify_countdown_expression(
                str(expression),
                numbers=sample.numbers,
                target=sample.target,
            )
            self.assertTrue(verification.is_valid)
            self.assertTrue(verification.reaches_target)


if __name__ == "__main__":
    unittest.main()

from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from pathlib import Path

from learning_to_reset.countdown_slices import is_hard_countdown_sample
from learning_to_reset.countdown_verifier import verify_countdown_expression
from learning_to_reset.data import load_countdown_samples
from learning_to_reset.synthetic_countdown_dataset import (
    build_sampling_profile,
    generate_synthetic_countdown_dataset,
    main,
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

    def test_generate_synthetic_countdown_dataset_can_require_hard_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            reference_path = Path(tmp_dir) / "reference.jsonl"
            output_path = Path(tmp_dir) / "generated-hard.jsonl"
            reference_path.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in [
                        {
                            "source_id": "ref-1",
                            "numbers": [6, 7, 8, 4],
                            "target": 2,
                            "question": "Use the numbers 6, 7, 8, 4 to reach 2.",
                        },
                        {
                            "source_id": "ref-2",
                            "numbers": [6, 7, 8, 4],
                            "target": 100,
                            "question": "Use the numbers 6, 7, 8, 4 to reach 100.",
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = generate_synthetic_countdown_dataset(
                reference_path=reference_path,
                output_path=output_path,
                num_samples=3,
                seed=11,
                require_hard=True,
            )
            generated_samples = load_countdown_samples(output_path)

        self.assertTrue(summary["require_hard"])
        self.assertEqual(len(generated_samples), 3)
        self.assertTrue(all(is_hard_countdown_sample(sample) for sample in generated_samples))
        self.assertTrue(all(sample.metadata["difficulty"] == "hard" for sample in generated_samples))

    def test_cli_accepts_require_hard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            reference_path = Path(tmp_dir) / "reference.jsonl"
            output_path = Path(tmp_dir) / "generated-hard.jsonl"
            reference_path.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in [
                        {
                            "source_id": "ref-1",
                            "numbers": [6, 7, 8, 4],
                            "target": 2,
                            "question": "Use the numbers 6, 7, 8, 4 to reach 2.",
                        },
                        {
                            "source_id": "ref-2",
                            "numbers": [6, 7, 8, 4],
                            "target": 100,
                            "question": "Use the numbers 6, 7, 8, 4 to reach 100.",
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(
                    [
                        "--reference-path",
                        str(reference_path),
                        "--output-path",
                        str(output_path),
                        "--num-samples",
                        "2",
                        "--seed",
                        "13",
                        "--require-hard",
                    ]
                )
            summary = json.loads(buffer.getvalue())
            generated_samples = load_countdown_samples(output_path)

        self.assertEqual(exit_code, 0)
        self.assertTrue(summary["require_hard"])
        self.assertEqual(len(generated_samples), 2)
        self.assertTrue(all(is_hard_countdown_sample(sample) for sample in generated_samples))


if __name__ == "__main__":
    unittest.main()

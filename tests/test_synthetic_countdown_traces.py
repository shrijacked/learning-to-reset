import tempfile
import unittest
from pathlib import Path

from learning_to_reset.countdown_solver import solve_countdown
from learning_to_reset.data import CountdownSample
from learning_to_reset.countdown_verifier import score_countdown_response
from learning_to_reset.synthetic_countdown_traces import (
    build_negative_trace_record,
    build_positive_trace_record,
    build_solution_walkthrough,
    build_synthetic_trace_records,
    generate_synthetic_trace_corpus,
)


class SyntheticCountdownTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = CountdownSample(
            numbers=(9, 11, 12, 17),
            target=70,
            question="Use the numbers 9, 11, 12, 17 to reach 70.",
            source_id="countdown-1",
        )

    def test_positive_trace_record_reaches_target(self) -> None:
        expression = solve_countdown(self.sample.numbers, self.sample.target)
        self.assertIsNotNone(expression)
        record = build_positive_trace_record(
            self.sample,
            solution_expression=str(expression),
        )

        verification = score_countdown_response(record["raw_trace"], self.sample)
        self.assertTrue(record["is_correct"])
        self.assertTrue(verification.is_valid)
        self.assertTrue(verification.reaches_target)
        self.assertIn("Step 1:", record["raw_trace"])
        self.assertIn("matches the target 70", record["raw_trace"])

    def test_negative_trace_record_stays_off_target(self) -> None:
        record = build_negative_trace_record(self.sample)

        verification = score_countdown_response(record["raw_trace"], self.sample)
        self.assertFalse(record["is_correct"])
        self.assertTrue(verification.is_valid)
        self.assertFalse(verification.reaches_target)
        self.assertIn("recovery_response", record)
        self.assertIn("evaluates to 9 instead of 70", record["raw_trace"])

        recovery_verification = score_countdown_response(record["recovery_response"], self.sample)
        self.assertTrue(recovery_verification.is_valid)
        self.assertTrue(recovery_verification.reaches_target)
        self.assertIn("After resetting the scratch work", record["recovery_response"])
        self.assertIn("Step 1:", record["recovery_response"])

    def test_build_solution_walkthrough_lists_intermediate_steps(self) -> None:
        walkthrough = build_solution_walkthrough(
            self.sample,
            solution_expression="((9 * 11) - (12 + 17))",
            after_reset=False,
        )

        self.assertIn("Compute (9 * 11) = 99.", walkthrough)
        self.assertIn("Compute (12 + 17) = 29.", walkthrough)
        self.assertIn("Compute ((9 * 11) - (12 + 17)) = 70.", walkthrough)

    def test_build_synthetic_trace_records_returns_paired_records(self) -> None:
        records, skipped = build_synthetic_trace_records([self.sample])

        self.assertEqual(skipped, 0)
        self.assertEqual(len(records), 2)
        self.assertTrue(records[0]["is_correct"])
        self.assertFalse(records[1]["is_correct"])

    def test_build_synthetic_trace_records_can_emit_multiple_solution_variants(self) -> None:
        records, skipped = build_synthetic_trace_records(
            [self.sample],
            solutions_per_sample=2,
        )

        self.assertEqual(skipped, 0)
        self.assertEqual(len(records), 4)
        positive_records = [record for record in records if record["is_correct"]]
        self.assertEqual(len(positive_records), 2)
        self.assertEqual(
            len({record["metadata"]["solution_expression"] for record in positive_records}),
            2,
        )

    def test_generate_synthetic_trace_corpus_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            countdown_path = Path(tmp_dir) / "countdown.jsonl"
            output_path = Path(tmp_dir) / "synthetic.jsonl"
            countdown_path.write_text(
                (
                    '{"source_id":"countdown-1","numbers":[9,11,12,17],'
                    '"target":70,"question":"Use the numbers 9, 11, 12, 17 to reach 70."}\n'
                ),
                encoding="utf-8",
            )

            summary = generate_synthetic_trace_corpus(
                countdown_path=countdown_path,
                output_path=output_path,
                solutions_per_sample=2,
            )
            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(summary["records_written"], 4)
        self.assertEqual(len(lines), 4)


if __name__ == "__main__":
    unittest.main()

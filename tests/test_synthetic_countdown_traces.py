import tempfile
import unittest
from pathlib import Path

from learning_to_reset.countdown_solver import solve_countdown
from learning_to_reset.data import CountdownSample
from learning_to_reset.countdown_verifier import score_countdown_response
from learning_to_reset.synthetic_countdown_traces import (
    build_contrastive_recovery_response,
    build_grounded_recovery_response,
    build_negative_trace_record,
    build_positive_trace_record,
    build_solution_walkthrough,
    build_synthetic_trace_records,
    build_verified_recovery_response,
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

    def test_verified_recovery_response_states_checked_expression_value(self) -> None:
        response = build_verified_recovery_response(
            self.sample,
            solution_expression="((9 * 11) - (12 + 17))",
        )

        verification = score_countdown_response(response, self.sample)
        self.assertTrue(verification.is_valid)
        self.assertTrue(verification.reaches_target)
        self.assertIn("Verifier check:", response)
        self.assertIn("= 70", response)
        self.assertIn("matches the target 70", response)

    def test_contrastive_recovery_response_rejects_failed_candidate(self) -> None:
        response = build_contrastive_recovery_response(
            self.sample,
            incorrect_expression="9",
            solution_expression="((9 * 11) - (12 + 17))",
        )

        verification = score_countdown_response(response, self.sample)
        self.assertTrue(verification.is_valid)
        self.assertTrue(verification.reaches_target)
        self.assertIn("Rejected candidate: 9 = 9, not 70.", response)
        self.assertIn(
            "Verified candidate: ((9 * 11) - (12 + 17)) = 70.",
            response,
        )

    def test_negative_trace_record_can_emit_contrastive_recovery(self) -> None:
        record = build_negative_trace_record(
            self.sample,
            recovery_style="contrastive",
        )

        recovery_verification = score_countdown_response(record["recovery_response"], self.sample)
        self.assertEqual(record["metadata"]["recovery_style"], "contrastive")
        self.assertTrue(recovery_verification.is_valid)
        self.assertTrue(recovery_verification.reaches_target)
        self.assertIn("Rejected candidate:", record["recovery_response"])
        self.assertIn("Verified candidate:", record["recovery_response"])

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

    def test_build_synthetic_trace_records_can_emit_both_recovery_styles(self) -> None:
        records, skipped = build_synthetic_trace_records(
            [self.sample],
            recovery_style="both",
        )

        self.assertEqual(skipped, 0)
        self.assertEqual(len(records), 3)
        recovery_responses = [
            record["recovery_response"]
            for record in records
            if not record["is_correct"]
        ]
        self.assertTrue(any("Step 1:" in response for response in recovery_responses))
        self.assertTrue(any("Verifier check:" in response for response in recovery_responses))

    def test_build_synthetic_trace_records_can_emit_all_recovery_styles(self) -> None:
        records, skipped = build_synthetic_trace_records(
            [self.sample],
            recovery_style="all",
        )

        self.assertEqual(skipped, 0)
        self.assertEqual(len(records), 5)
        recovery_responses = [
            record["recovery_response"]
            for record in records
            if not record["is_correct"]
        ]
        self.assertTrue(any("Step 1:" in response for response in recovery_responses))
        self.assertTrue(any("Verifier check:" in response for response in recovery_responses))
        self.assertTrue(any("Rejected candidate:" in response for response in recovery_responses))
        self.assertTrue(any("Number budget" in response for response in recovery_responses))

    def test_grounded_recovery_response_includes_substep_arithmetic(self) -> None:
        response = build_grounded_recovery_response(
            self.sample,
            solution_expression="((9 * 11) - (12 + 17))",
        )

        verification = score_countdown_response(response, self.sample)
        self.assertTrue(verification.is_valid)
        self.assertTrue(verification.reaches_target)
        self.assertIn("Compute (9 * 11) = 99.", response)
        self.assertIn("Compute (12 + 17) = 29.", response)
        self.assertIn("Compute ((9 * 11) - (12 + 17)) = 70.", response)

    def test_grounded_recovery_response_lists_rejected_hypothesis_with_substeps(
        self,
    ) -> None:
        response = build_grounded_recovery_response(
            self.sample,
            solution_expression="((9 * 11) - (12 + 17))",
            rejected_expression="(11 + 12)",
        )

        self.assertIn("(11 + 12)", response)
        self.assertIn("Compute (11 + 12) = 23.", response)
        self.assertIn("23", response)
        self.assertIn("Reject", response)

    def test_grounded_recovery_response_includes_number_budget_audit(self) -> None:
        response = build_grounded_recovery_response(
            self.sample,
            solution_expression="((9 * 11) - (12 + 17))",
        )
        for number in self.sample.numbers:
            self.assertIn(f"used {number}", response)
        self.assertIn("4 of 4", response)

    def test_grounded_recovery_response_final_value_matches_target(self) -> None:
        response = build_grounded_recovery_response(
            self.sample,
            solution_expression="((9 * 11) - (12 + 17))",
        )
        self.assertIn("matches target 70", response)

    def test_grounded_recovery_response_seeds_template_variation(self) -> None:
        base = build_grounded_recovery_response(
            self.sample,
            solution_expression="((9 * 11) - (12 + 17))",
            rng_seed=0,
        )
        other = build_grounded_recovery_response(
            self.sample,
            solution_expression="((9 * 11) - (12 + 17))",
            rng_seed=7,
        )
        self.assertNotEqual(base, other)
        for response in (base, other):
            verification = score_countdown_response(response, self.sample)
            self.assertTrue(verification.is_valid)
            self.assertTrue(verification.reaches_target)

    def test_negative_trace_record_can_emit_grounded_recovery(self) -> None:
        record = build_negative_trace_record(
            self.sample,
            recovery_style="grounded",
        )

        recovery_verification = score_countdown_response(
            record["recovery_response"], self.sample
        )
        self.assertEqual(record["metadata"]["recovery_style"], "grounded")
        self.assertTrue(recovery_verification.is_valid)
        self.assertTrue(recovery_verification.reaches_target)
        self.assertIn("Number budget", record["recovery_response"])
        self.assertIn("Reject", record["recovery_response"])

    def test_build_synthetic_trace_records_can_emit_grounded_recovery_only(
        self,
    ) -> None:
        records, skipped = build_synthetic_trace_records(
            [self.sample],
            recovery_style="grounded",
        )

        self.assertEqual(skipped, 0)
        self.assertEqual(len(records), 2)
        negative_records = [r for r in records if not r["is_correct"]]
        self.assertEqual(len(negative_records), 1)
        self.assertEqual(
            negative_records[0]["metadata"]["recovery_style"], "grounded"
        )
        self.assertIn("Number budget", negative_records[0]["recovery_response"])

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
                recovery_style="both",
            )
            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(summary["records_written"], 6)
        self.assertEqual(summary["recovery_style"], "both")
        self.assertEqual(len(lines), 6)


if __name__ == "__main__":
    unittest.main()

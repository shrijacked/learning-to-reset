import tempfile
import unittest
from pathlib import Path

from learning_to_reset.countdown_verifier import score_countdown_response
from learning_to_reset.failure_recovery_traces import (
    build_failure_recovery_trace_records,
    generate_failure_recovery_trace_corpus,
)
from learning_to_reset.prompts import PromptExample, build_reasoning_prompt


class FailureRecoveryTraceTests(unittest.TestCase):
    def test_build_failure_recovery_trace_records_solves_failed_hard_eval(self) -> None:
        example = PromptExample(
            prompt=build_reasoning_prompt(
                "Use the numbers 22, 72, 19, 25 to reach 74.",
                allow_clean=True,
            ),
            response="",
            metadata={
                "source_id": "hard-1",
                "numbers": (22, 72, 19, 25),
                "target": 74,
                "question": "Use the numbers 22, 72, 19, 25 to reach 74.",
            },
        )
        failed_result = {
            "source_id": "hard-1",
            "response": "<think>I got stuck and should reset.</think><clean>",
            "is_valid": False,
            "reaches_target": False,
            "reason": "No answer block.",
        }

        records, summary = build_failure_recovery_trace_records(
            [example],
            [failed_result],
            recovery_style="contrastive",
        )

        self.assertEqual(summary["failures_seen"], 1)
        self.assertEqual(summary["records_written"], 1)
        self.assertEqual(summary["skipped_solved_or_valid"], 0)
        self.assertEqual(records[0]["metadata"]["source"], "failure_mined_countdown_solver")
        self.assertFalse(records[0]["is_correct"])
        self.assertIn("<clean>", records[0]["raw_trace"])

        verification = score_countdown_response(
            records[0]["recovery_response"],
            sample=type(
                "Sample",
                (),
                {
                    "numbers": (22, 72, 19, 25),
                    "target": 74,
                    "source_id": "hard-1",
                },
            )(),
        )
        self.assertTrue(verification.is_valid)
        self.assertTrue(verification.reaches_target)

    def test_build_failure_recovery_trace_records_skips_correct_results(self) -> None:
        example = PromptExample(
            prompt="Question: Reach 24.",
            response="",
            metadata={
                "source_id": "easy-1",
                "numbers": (9, 8, 3, 1),
                "target": 24,
                "question": "Reach 24.",
            },
        )
        correct_result = {
            "source_id": "easy-1",
            "response": "<think>Done.</think><answer>8 * 3</answer>",
            "is_valid": True,
            "reaches_target": True,
        }

        records, summary = build_failure_recovery_trace_records([example], [correct_result])

        self.assertEqual(records, [])
        self.assertEqual(summary["records_written"], 0)
        self.assertEqual(summary["skipped_solved_or_valid"], 1)

    def test_grounded_recovery_style_propagates_through_failure_mining(self) -> None:
        example = PromptExample(
            prompt=build_reasoning_prompt(
                "Use the numbers 22, 72, 19, 25 to reach 74.",
                allow_clean=True,
            ),
            response="",
            metadata={
                "source_id": "hard-2",
                "numbers": (22, 72, 19, 25),
                "target": 74,
                "question": "Use the numbers 22, 72, 19, 25 to reach 74.",
            },
        )
        failed_result = {
            "source_id": "hard-2",
            "response": "<think>I got stuck.</think><clean>",
            "is_valid": False,
            "reaches_target": False,
            "reason": "No answer block.",
        }

        records, summary = build_failure_recovery_trace_records(
            [example],
            [failed_result],
            recovery_style="grounded",
        )

        self.assertEqual(summary["records_written"], 1)
        self.assertEqual(records[0]["metadata"]["recovery_style"], "grounded")
        self.assertIn("Number budget", records[0]["recovery_response"])

        sample = type(
            "Sample",
            (),
            {
                "numbers": (22, 72, 19, 25),
                "target": 74,
                "source_id": "hard-2",
            },
        )()
        verification = score_countdown_response(records[0]["recovery_response"], sample)
        self.assertTrue(verification.is_valid)
        self.assertTrue(verification.reaches_target)

    def test_build_failure_recovery_trace_records_rejects_eval_overlap(self) -> None:
        example = PromptExample(
            prompt=build_reasoning_prompt(
                "Use the numbers 22, 72, 19, 25 to reach 74.",
                allow_clean=True,
            ),
            response="",
            metadata={
                "source_id": "hard-1",
                "numbers": (22, 72, 19, 25),
                "target": 74,
                "question": "Use the numbers 22, 72, 19, 25 to reach 74.",
            },
        )
        failed_result = {
            "source_id": "hard-1",
            "response": "<think>I got stuck and should reset.</think><clean>",
            "is_valid": False,
            "reaches_target": False,
            "reason": "No answer block.",
        }

        with self.assertRaises(ValueError) as raised:
            build_failure_recovery_trace_records(
                [example],
                [failed_result],
                recovery_style="contrastive",
                excluded_source_ids={"hard-1", "hard-9"},
            )

        self.assertIn("contamination", str(raised.exception).lower())
        self.assertIn("hard-1", str(raised.exception))

    def test_build_failure_recovery_trace_records_passes_disjoint_exclusion(self) -> None:
        example = PromptExample(
            prompt=build_reasoning_prompt(
                "Use the numbers 22, 72, 19, 25 to reach 74.",
                allow_clean=True,
            ),
            response="",
            metadata={
                "source_id": "hard-3",
                "numbers": (22, 72, 19, 25),
                "target": 74,
                "question": "Use the numbers 22, 72, 19, 25 to reach 74.",
            },
        )
        failed_result = {
            "source_id": "hard-3",
            "response": "<think>Stuck.</think><clean>",
            "is_valid": False,
            "reaches_target": False,
        }

        records, summary = build_failure_recovery_trace_records(
            [example],
            [failed_result],
            recovery_style="contrastive",
            excluded_source_ids={"holdout-1", "holdout-2"},
        )

        self.assertEqual(summary["records_written"], 1)
        self.assertEqual(summary["excluded_source_ids"], 2)

    def test_generate_failure_recovery_trace_corpus_loads_exclusion_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            prepared_path = root / "countdown-train-hard.jsonl"
            results_path = root / "results.jsonl"
            holdout_path = root / "countdown-test-hard.jsonl"
            output_path = root / "mined-recoveries.jsonl"
            prepared_path.write_text(
                (
                    '{"prompt":"Question: Use 87, 49, 91, 31 to reach 13.",'
                    '"response":"","metadata":{"source_id":"hard-X",'
                    '"numbers":[87,49,91,31],"target":13,'
                    '"question":"Use 87, 49, 91, 31 to reach 13."}}\n'
                ),
                encoding="utf-8",
            )
            holdout_path.write_text(
                (
                    '{"prompt":"Question: held out","response":"",'
                    '"metadata":{"source_id":"hard-X","numbers":[1,2,3,4],'
                    '"target":10,"question":"held out"}}\n'
                ),
                encoding="utf-8",
            )
            results_path.write_text(
                (
                    '{"source_id":"hard-X","response":"<think>Bad.</think><clean>",'
                    '"is_valid":false,"reaches_target":false}\n'
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                generate_failure_recovery_trace_corpus(
                    prepared_countdown_path=prepared_path,
                    eval_results_path=results_path,
                    output_path=output_path,
                    recovery_style="verification",
                    exclude_source_id_paths=[holdout_path],
                )

    def test_generate_failure_recovery_trace_corpus_writes_jsonl_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            prepared_path = root / "countdown-test-hard.jsonl"
            results_path = root / "results.jsonl"
            output_path = root / "mined-recoveries.jsonl"
            prepared_path.write_text(
                (
                    '{"prompt":"Question: Use the numbers 87, 49, 91, 31 to reach 13.",'
                    '"response":"","metadata":{"source_id":"hard-2",'
                    '"numbers":[87,49,91,31],"target":13,'
                    '"question":"Use the numbers 87, 49, 91, 31 to reach 13."}}\n'
                ),
                encoding="utf-8",
            )
            results_path.write_text(
                (
                    '{"source_id":"hard-2","response":"<think>Bad path.</think><clean>",'
                    '"is_valid":false,"reaches_target":false}\n'
                ),
                encoding="utf-8",
            )

            summary = generate_failure_recovery_trace_corpus(
                prepared_countdown_path=prepared_path,
                eval_results_path=results_path,
                output_path=output_path,
                recovery_style="verification",
            )
            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(summary["records_written"], 1)
        self.assertEqual(summary["recovery_style"], "verification")
        self.assertEqual(len(lines), 1)


if __name__ == "__main__":
    unittest.main()

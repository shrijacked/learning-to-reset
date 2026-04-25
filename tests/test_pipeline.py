import unittest

from learning_to_reset.data import CountdownSample, TraceRecord
from learning_to_reset.pipeline import (
    DatasetSplit,
    batch_prompt_examples,
    prepare_countdown_examples,
    prepare_sft_examples,
    split_sequence,
)


class PipelineTests(unittest.TestCase):
    def test_prepare_sft_examples_uses_curated_responses(self) -> None:
        records = (
            TraceRecord(
                source_id="trace-1",
                problem="Make 10 from 7, 2, 1",
                raw_trace="<think>Valid.</think><answer>10</answer>",
                is_correct=True,
            ),
            TraceRecord(
                source_id="trace-2",
                problem="Make 9 from 6, 2, 1",
                raw_trace="<think>Wrong turn.</think><answer>8</answer>",
                is_correct=False,
            ),
        )

        examples = prepare_sft_examples(records)

        self.assertEqual(len(examples), 2)
        self.assertIn("<answer>", examples[0].response)
        self.assertTrue(examples[1].response.strip().endswith("<clean>"))
        self.assertEqual(examples[1].metadata["source_id"], "trace-2")

    def test_prepare_sft_examples_can_append_retry_recovery_examples(self) -> None:
        records = (
            TraceRecord(
                source_id="trace-1",
                problem="Make 10 from 7, 2, 1",
                raw_trace="<think>Wrong turn.</think><answer>9</answer>",
                is_correct=False,
                metadata={
                    "recovery_response": (
                        "<think>Fresh start.</think><answer>(7 + 2) + 1</answer>"
                    )
                },
            ),
        )

        examples = prepare_sft_examples(records, include_recovery_examples=True)

        self.assertEqual(len(examples), 2)
        self.assertIn("<clean>", examples[0].prompt)
        self.assertNotIn("<clean>", examples[1].prompt)
        self.assertIn("<answer>", examples[1].response)
        self.assertEqual(examples[1].metadata["stage"], "retry-recovery")

    def test_prepare_sft_examples_can_repeat_retry_recovery_examples(self) -> None:
        records = (
            TraceRecord(
                source_id="trace-1",
                problem="Make 10 from 7, 2, 1",
                raw_trace="<think>Wrong turn.</think><answer>9</answer>",
                is_correct=False,
                metadata={
                    "recovery_response": (
                        "<think>Fresh start.</think><answer>(7 + 2) + 1</answer>"
                    )
                },
            ),
        )

        examples = prepare_sft_examples(
            records,
            include_recovery_examples=True,
            recovery_repeat=3,
        )

        self.assertEqual(len(examples), 4)
        self.assertEqual(sum(int(example.metadata.get("stage") == "retry-recovery") for example in examples), 3)

    def test_prepare_sft_examples_can_require_target_correct_recovery(self) -> None:
        records = (
            TraceRecord(
                source_id="trace-good",
                problem="Make 10 from 7, 2, 1",
                raw_trace="<think>Wrong turn.</think><answer>9</answer>",
                is_correct=False,
                metadata={
                    "numbers": [7, 2, 1],
                    "target": 10,
                    "recovery_response": (
                        "<think>Fresh start.</think><answer>((7 + 2) + 1)</answer>"
                    ),
                },
            ),
            TraceRecord(
                source_id="trace-bad",
                problem="Make 10 from 7, 2, 1",
                raw_trace="<think>Wrong turn.</think><answer>9</answer>",
                is_correct=False,
                metadata={
                    "numbers": [7, 2, 1],
                    "target": 10,
                    "recovery_response": (
                        "<think>Still wrong.</think><answer>(7 + 2)</answer>"
                    ),
                },
            ),
        )

        examples = prepare_sft_examples(
            records,
            include_recovery_examples=True,
            require_recovery_target_correct=True,
        )

        recovery_examples = [
            example
            for example in examples
            if example.metadata.get("stage") == "retry-recovery"
        ]
        self.assertEqual(len(examples), 3)
        self.assertEqual(len(recovery_examples), 1)
        self.assertEqual(recovery_examples[0].metadata["source_id"], "trace-good")
        self.assertEqual(recovery_examples[0].metadata["recovery_expression"], "((7 + 2) + 1)")

    def test_require_target_correct_recovery_drops_inconsistent_arithmetic_claims(
        self,
    ) -> None:
        records = (
            TraceRecord(
                source_id="trace-bad-claim",
                problem="Make 10 from 7, 2, 1",
                raw_trace="<think>Wrong turn.</think><answer>9</answer>",
                is_correct=False,
                metadata={
                    "numbers": [7, 2, 1],
                    "target": 10,
                    "recovery_response": (
                        "<think>2 + 1 = 7.</think><answer>((7 + 2) + 1)</answer>"
                    ),
                },
            ),
            TraceRecord(
                source_id="trace-good-claim",
                problem="Make 10 from 7, 2, 1",
                raw_trace="<think>Wrong turn.</think><answer>9</answer>",
                is_correct=False,
                metadata={
                    "numbers": [7, 2, 1],
                    "target": 10,
                    "recovery_response": (
                        "<think>Compute (7 + 2) = 9.</think>"
                        "<answer>((7 + 2) + 1)</answer>"
                    ),
                },
            ),
        )

        examples = prepare_sft_examples(
            records,
            include_recovery_examples=True,
            require_recovery_target_correct=True,
        )

        recovery_examples = [
            example
            for example in examples
            if example.metadata.get("stage") == "retry-recovery"
        ]
        self.assertEqual(len(recovery_examples), 1)
        self.assertEqual(
            recovery_examples[0].metadata["source_id"], "trace-good-claim"
        )

    def test_target_correct_recovery_gate_skips_missing_countdown_metadata(self) -> None:
        records = (
            TraceRecord(
                source_id="trace-missing-metadata",
                problem="Make 10 from 7, 2, 1",
                raw_trace="<think>Wrong turn.</think><answer>9</answer>",
                is_correct=False,
                metadata={
                    "recovery_response": (
                        "<think>Fresh start.</think><answer>((7 + 2) + 1)</answer>"
                    ),
                },
            ),
        )

        examples = prepare_sft_examples(
            records,
            include_recovery_examples=True,
            require_recovery_target_correct=True,
        )

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].metadata["source_id"], "trace-missing-metadata")

    def test_prepare_countdown_examples_carries_target_metadata(self) -> None:
        samples = (
            CountdownSample(
                source_id="sample-1",
                numbers=(25, 7, 3, 2),
                target=50,
                question="Use the numbers 25, 7, 3, 2 to reach 50.",
            ),
        )

        examples = prepare_countdown_examples(samples, allow_clean=True)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].response, "")
        self.assertEqual(examples[0].metadata["target"], 50)
        self.assertEqual(examples[0].metadata["numbers"], (25, 7, 3, 2))

    def test_batch_prompt_examples_chunks_examples(self) -> None:
        records = (
            TraceRecord(
                source_id="trace-1",
                problem="P1",
                raw_trace="<think>A</think><answer>1</answer>",
                is_correct=True,
            ),
            TraceRecord(
                source_id="trace-2",
                problem="P2",
                raw_trace="<think>B</think><answer>2</answer>",
                is_correct=True,
            ),
            TraceRecord(
                source_id="trace-3",
                problem="P3",
                raw_trace="<think>C</think><answer>3</answer>",
                is_correct=True,
            ),
        )

        batches = batch_prompt_examples(prepare_sft_examples(records), batch_size=2)

        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0].examples), 2)
        self.assertEqual(len(batches[1].examples), 1)

    def test_split_sequence_is_deterministic_and_order_preserving(self) -> None:
        split = split_sequence(tuple(range(10)), train_ratio=0.6, val_ratio=0.2)

        self.assertIsInstance(split, DatasetSplit)
        self.assertEqual(split.train, (0, 1, 2, 3, 4, 5))
        self.assertEqual(split.validation, (6, 7))
        self.assertEqual(split.test, (8, 9))


if __name__ == "__main__":
    unittest.main()

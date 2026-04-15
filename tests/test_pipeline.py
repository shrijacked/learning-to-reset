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

import json
import tempfile
import unittest
from pathlib import Path

from learning_to_reset.data import load_countdown_samples, load_trace_records
from learning_to_reset.prompts import (
    build_countdown_prompt,
    build_reasoning_prompt,
    build_sft_training_example,
    parse_reasoning_prompt,
)


class DataPipelineTests(unittest.TestCase):
    def test_load_trace_records_supports_jsonl_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "traces.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "trace-1",
                                "question": "Make 10 from 7, 2, 1",
                                "trace": "<think>Try something.</think><answer>10</answer>",
                                "correct": True,
                            }
                        ),
                        json.dumps(
                            {
                                "id": "trace-2",
                                "prompt": "Make 12 from 8, 3, 1",
                                "raw_trace": "<think>Bad path.</think><answer>11</answer>",
                                "is_correct": False,
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            records = load_trace_records(path)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].source_id, "trace-1")
        self.assertEqual(records[1].problem, "Make 12 from 8, 3, 1")
        self.assertFalse(records[1].is_correct)

    def test_load_trace_records_flattens_metadata_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "traces.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "trace-1",
                        "question": "Make 10 from 7, 2, 1",
                        "trace": "<think>Try something.</think><answer>10</answer>",
                        "correct": True,
                        "metadata": {"source_dataset": "paper", "row_index": 3},
                        "bootstrap_kind": "think_only_negative",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            record = load_trace_records(path)[0]

        self.assertEqual(record.metadata["source_dataset"], "paper")
        self.assertEqual(record.metadata["row_index"], 3)
        self.assertEqual(record.metadata["bootstrap_kind"], "think_only_negative")
        self.assertNotIn("metadata", record.metadata)

    def test_load_countdown_samples_renders_question_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "countdown.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "id": "sample-1",
                            "numbers": [25, 7, 3, 2],
                            "target": 50,
                            "solution": "(25 * 2)",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            samples = load_countdown_samples(path)

        self.assertEqual(len(samples), 1)
        self.assertIn("25, 7, 3, 2", samples[0].question)
        self.assertIn("50", samples[0].question)

    def test_load_countdown_samples_flattens_metadata_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "countdown.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "sample-1",
                        "numbers": [25, 7, 3, 2],
                        "target": 50,
                        "metadata": {"source_dataset": "countdown-env", "row_index": 0},
                        "difficulty": "hard",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            sample = load_countdown_samples(path)[0]

        self.assertEqual(sample.metadata["source_dataset"], "countdown-env")
        self.assertEqual(sample.metadata["row_index"], 0)
        self.assertEqual(sample.metadata["difficulty"], "hard")
        self.assertNotIn("metadata", sample.metadata)

    def test_build_reasoning_prompt_handles_clean_flag(self) -> None:
        with_clean = build_reasoning_prompt("Solve this problem", allow_clean=True)
        without_clean = build_reasoning_prompt("Solve this problem", allow_clean=False)

        self.assertIn("<clean>", with_clean)
        self.assertNotIn("<clean>", without_clean)
        self.assertIn("Solve this problem", with_clean)

    def test_build_sft_training_example_curates_incorrect_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "traces.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "id": "trace-3",
                            "problem": "Make 9 from 6, 2, 1",
                            "trace": "<think>Wrong turn.</think><answer>8</answer>",
                            "is_correct": False,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            record = load_trace_records(path)[0]

        example = build_sft_training_example(record)

        self.assertIn("Make 9 from 6, 2, 1", example.prompt)
        self.assertTrue(example.response.strip().endswith("<clean>"))
        self.assertTrue(example.metadata["uses_clean"])

    def test_build_countdown_prompt_uses_sample_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "countdown.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "sample-2",
                        "numbers": "9 8 3 1",
                        "target": "24",
                        "question": "Reach 24 using 9, 8, 3, 1.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            sample = load_countdown_samples(path)[0]

        prompt = build_countdown_prompt(sample, allow_clean=True)

        self.assertIn("Reach 24 using 9, 8, 3, 1.", prompt)
        self.assertIn("<clean>", prompt)

    def test_parse_reasoning_prompt_recovers_question_and_clean_instructions(self) -> None:
        prompt = build_reasoning_prompt("Reach 68 using 60, 27, 19.", allow_clean=True)

        parsed = parse_reasoning_prompt(prompt)

        self.assertIn("<answer>", parsed.base_instructions)
        self.assertIsNotNone(parsed.clean_instructions)
        self.assertIn("<clean>", parsed.clean_instructions or "")
        self.assertEqual(parsed.question, "Reach 68 using 60, 27, 19.")


if __name__ == "__main__":
    unittest.main()

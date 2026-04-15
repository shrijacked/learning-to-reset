import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from learning_to_reset.paper_sources import (
    expand_countdown_hf_row,
    expand_reference_trace_hf_row,
    extract_behavior_question,
    fetch_paper_source_bundle,
    fetch_reference_trace_records,
    parse_countdown_user_prompt,
    write_jsonl_records,
)


class PaperSourcesTests(unittest.TestCase):
    def test_extract_behavior_question_uses_user_block(self) -> None:
        query = (
            "A conversation between User and Assistant.\n"
            "User: Solve x + 2 = 5. Show your work in <think> tags.\n"
            "Assistant: Let me solve this step by step.\n"
        )

        question = extract_behavior_question(query)

        self.assertEqual(question, "Solve x + 2 = 5. Show your work in <think> tags.")

    def test_parse_countdown_user_prompt_recovers_numbers_and_target(self) -> None:
        numbers, target = parse_countdown_user_prompt(
            "Using the numbers [9, 11, 12, 17], create an equation that equals 70."
        )

        self.assertEqual(numbers, (9, 11, 12, 17))
        self.assertEqual(target, 70)

    def test_expand_countdown_hf_row_flattens_batched_metadata(self) -> None:
        row = {
            "prompt": [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": "Using the numbers [9, 11, 12, 17], create an equation that equals 70.",
                },
            ],
            "metadata": {
                "T_max": 5,
                "numbers": [[9, 11, 12, 17], [15, 3, 43, 18]],
                "target": [70, 34],
            },
        }

        records = expand_countdown_hf_row(row, dataset_id="countdown-env", split="train", row_index=3)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["numbers"], [9, 11, 12, 17])
        self.assertEqual(records[0]["target"], 70)
        self.assertEqual(
            records[0]["question"],
            "Using the numbers [9, 11, 12, 17], create an equation that equals 70.",
        )
        self.assertEqual(records[1]["numbers"], [15, 3, 43, 18])
        self.assertEqual(records[1]["target"], 34)
        self.assertIn("15, 3, 43, 18", records[1]["question"])
        self.assertEqual(records[1]["source_id"], "countdown-env:train:3:1")

    def test_expand_reference_trace_hf_row_extracts_question_and_completion(self) -> None:
        row = {
            "query": (
                "A conversation between User and Assistant.\n"
                "User: Solve x + 2 = 5. Show your work in <think> tags.\n"
                "Assistant: Let me solve this step by step.\n"
            ),
            "completion": "<think>Subtract 2 from both sides.</think><answer>x = 3</answer>",
        }

        record = expand_reference_trace_hf_row(
            row,
            dataset_id="owm-cog-behaviors",
            split="train",
            row_index=7,
        )

        self.assertEqual(record["problem"], "Solve x + 2 = 5. Show your work in <think> tags.")
        self.assertEqual(
            record["raw_trace"],
            "<think>Subtract 2 from both sides.</think><answer>x = 3</answer>",
        )
        self.assertTrue(record["is_correct"])
        self.assertEqual(record["source_id"], "owm-cog-behaviors:train:7")
        self.assertEqual(record["metadata"]["source_dataset"], "owm-cog-behaviors")

    def test_fetch_reference_trace_records_writes_positive_trace_jsonl(self) -> None:
        rows = [
            {
                "query": (
                    "A conversation between User and Assistant.\n"
                    "User: Solve x + 2 = 5. Show your work in <think> tags.\n"
                    "Assistant: Let me solve this step by step.\n"
                ),
                "completion": "<think>Subtract 2 from both sides.</think><answer>x = 3</answer>",
            },
            {
                "query": (
                    "A conversation between User and Assistant.\n"
                    "User: Compute 2 + 2. Show your work in <think> tags.\n"
                    "Assistant: Let me solve this step by step.\n"
                ),
                "completion": "<think>Add the integers.</think><answer>4</answer>",
            },
        ]

        def fake_load_dataset(dataset_id: str, *, split: str, streaming: bool):
            self.assertEqual(dataset_id, "obiwan96/owm-cog-behaviors")
            self.assertEqual(split, "train")
            self.assertTrue(streaming)
            return iter(rows)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "reference-positive-traces.jsonl"
            with patch("learning_to_reset.paper_sources._require_datasets", return_value=fake_load_dataset):
                summary = fetch_reference_trace_records(
                    dataset_id="obiwan96/owm-cog-behaviors",
                    split="train",
                    output_path=output,
                )

            payload = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(summary["records_written"], 2)
        self.assertEqual(len(payload), 2)
        self.assertIn('"problem": "Solve x + 2 = 5. Show your work in <think> tags."', payload[0])
        self.assertIn('"is_correct": true', payload[0])

    def test_fetch_paper_source_bundle_writes_bootstrapped_trace_corpus(self) -> None:
        dataset_rows = {
            "train-ds": [
                {
                    "prompt": [
                        {"role": "system", "content": "system"},
                        {
                            "role": "user",
                            "content": "Using the numbers [9, 11, 12, 17], create an equation that equals 70.",
                        },
                    ],
                    "metadata": {
                        "T_max": 5,
                        "numbers": [[9, 11, 12, 17]],
                        "target": [70],
                    },
                }
            ],
            "eval-ds": [
                {
                    "prompt": [
                        {"role": "system", "content": "system"},
                        {
                            "role": "user",
                            "content": "Using the numbers [15, 3, 43, 18], create an equation that equals 34.",
                        },
                    ],
                    "metadata": {
                        "T_max": 5,
                        "numbers": [[15, 3, 43, 18]],
                        "target": [34],
                    },
                }
            ],
            "trace-ds": [
                {
                    "query": (
                        "A conversation between User and Assistant.\n"
                        "User: Solve x + 2 = 5. Show your work in <think> tags.\n"
                        "Assistant: Let me solve this step by step.\n"
                    ),
                    "completion": "<think>Subtract 2 from both sides.</think><answer>x = 3</answer>",
                }
            ],
        }

        def fake_load_dataset(dataset_id: str, *, split: str, streaming: bool):
            self.assertEqual(split, "train" if dataset_id != "eval-ds" else "eval")
            self.assertTrue(streaming)
            return iter(dataset_rows[dataset_id])

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "paper-assets"
            with patch("learning_to_reset.paper_sources._require_datasets", return_value=fake_load_dataset):
                summary = fetch_paper_source_bundle(
                    output_dir=output_dir,
                    countdown_train_dataset="train-ds",
                    countdown_eval_dataset="eval-ds",
                    reference_trace_dataset="trace-ds",
                )

            positive_path = output_dir / "reference-positive-traces.jsonl"
            paired_path = output_dir / "reference-traces.jsonl"
            positive_payload = positive_path.read_text(encoding="utf-8").splitlines()
            paired_payload = paired_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(summary["countdown_train"]["samples_written"], 1)
        self.assertEqual(summary["countdown_eval"]["samples_written"], 1)
        self.assertEqual(summary["reference_traces"]["records_written"], 1)
        self.assertEqual(summary["bootstrapped_reference_traces"]["records_written"], 2)
        self.assertEqual(len(positive_payload), 1)
        self.assertEqual(len(paired_payload), 2)
        self.assertIn('"is_correct": false', paired_payload[1])

    def test_write_jsonl_records_uses_ascii_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "records.jsonl"
            write_jsonl_records(
                [
                    {"source_id": "x1", "target": 70},
                    {"source_id": "x2", "target": 34},
                ],
                output,
            )

            payload = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(payload), 2)
        self.assertIn('"source_id": "x1"', payload[0])


if __name__ == "__main__":
    unittest.main()

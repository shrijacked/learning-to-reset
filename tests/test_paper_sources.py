import tempfile
import unittest
from pathlib import Path

from learning_to_reset.paper_sources import (
    expand_countdown_hf_row,
    extract_behavior_question,
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

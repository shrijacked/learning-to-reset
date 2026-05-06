import json
import tempfile
import unittest
from pathlib import Path

from learning_to_reset.sft_runtime import (
    PromptCollator,
    load_prepared_examples,
    render_training_text,
    tokenize_training_example,
)
from learning_to_reset.prompts import PromptExample


class DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 99
        self._vocab = {}
        self._next_id = 1

    def encode(self, text, add_special_tokens=False):
        ids = []
        for token in text.split():
            if token not in self._vocab:
                self._vocab[token] = self._next_id
                self._next_id += 1
            ids.append(self._vocab[token])
        return ids


class SFTRuntimeTests(unittest.TestCase):
    @staticmethod
    def _materialize_batch_values(values):
        if hasattr(values, "tolist"):
            return values.tolist()
        return values

    def test_load_prepared_examples_reads_jsonl_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "prepared.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "prompt": "Question: example",
                        "response": "<answer>42</answer>",
                        "metadata": {"source_id": "x1"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            examples = load_prepared_examples(path)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].metadata["source_id"], "x1")

    def test_load_prepared_examples_max_examples_truncates(self) -> None:
        rows = [
            {"prompt": f"p{i}", "response": f"r{i}", "metadata": {"i": i}}
            for i in range(5)
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "prepared.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            capped = load_prepared_examples(path, max_examples=2)
        self.assertEqual(len(capped), 2)
        self.assertEqual(capped[0].metadata["i"], 0)
        self.assertEqual(capped[1].metadata["i"], 1)

    def test_load_prepared_examples_rejects_missing_prompt_in_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "prepared.jsonl"
            path.write_text(
                json.dumps({"response": "<answer>42</answer>"}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "row 1.*prompt"):
                load_prepared_examples(path)

    def test_load_prepared_examples_rejects_extra_top_level_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "prepared.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "prompt": "Question: example",
                        "response": "<answer>42</answer>",
                        "metadata": {},
                        "raw_trace": "<answer>42</answer>",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unexpected top-level keys"):
                load_prepared_examples(path)

    def test_render_training_text_combines_prompt_and_response(self) -> None:
        example = PromptExample(prompt="Question: solve", response="<answer>42</answer>")

        rendered = render_training_text(example)

        self.assertIn("Question: solve", rendered)
        self.assertIn("<answer>42</answer>", rendered)

    def test_tokenize_training_example_masks_prompt_tokens(self) -> None:
        tokenizer = DummyTokenizer()
        example = PromptExample(prompt="Question solve", response="<answer>42</answer>")

        features = tokenize_training_example(example, tokenizer, max_length=32)

        self.assertEqual(len(features["input_ids"]), len(features["labels"]))
        self.assertEqual(features["labels"][0], -100)
        self.assertEqual(features["labels"][1], -100)
        self.assertNotEqual(features["labels"][-1], -100)

    def test_prompt_collator_pads_labels_with_negative_hundreds(self) -> None:
        collator = PromptCollator(pad_token_id=0)
        batch = collator(
            [
                {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
                {"input_ids": [4], "attention_mask": [1], "labels": [-100]},
            ]
        )

        self.assertEqual(
            self._materialize_batch_values(batch["input_ids"]),
            [[1, 2, 3], [4, 0, 0]],
        )
        self.assertEqual(
            self._materialize_batch_values(batch["labels"]),
            [[-100, 2, 3], [-100, -100, -100]],
        )


if __name__ == "__main__":
    unittest.main()

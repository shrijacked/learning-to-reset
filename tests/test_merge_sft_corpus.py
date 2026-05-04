import json
import tempfile
import unittest
from pathlib import Path

from learning_to_reset.merge_sft_corpus import merge_sft_train_with_mined_traces
from learning_to_reset.sft_runtime import load_prepared_examples


class MergeSftCorpusTests(unittest.TestCase):
    def test_merge_converts_mined_traces_to_prompt_examples(self) -> None:
        base_line = {
            "prompt": "Question: demo",
            "response": "<think>x</think><answer>1</answer>",
            "metadata": {"source_id": "base-1"},
        }
        mined_line = {
            "source_id": "mined-src",
            "problem": "Reach 2 using 1, 1.",
            "raw_trace": (
                "<think>try</think>\n"
                "<answer>1+1</answer>"
            ),
            "recovery_response": (
                "<think>fixed</think>\n<answer>1+1</answer>"
            ),
            "is_correct": False,
            "metadata": {"numbers": [1, 1], "target": 2},
        }
        with tempfile.TemporaryDirectory(prefix="ltr-merge-sft-") as tmp:
            base_path = Path(tmp) / "sft.jsonl"
            mined_path = Path(tmp) / "mined.jsonl"
            out_path = Path(tmp) / "out.jsonl"
            base_path.write_text(json.dumps(base_line) + "\n", encoding="utf-8")
            mined_path.write_text(json.dumps(mined_line) + "\n", encoding="utf-8")
            b, m, t = merge_sft_train_with_mined_traces(
                sft_train_path=base_path,
                mined_traces_path=mined_path,
                output_path=out_path,
            )
            self.assertEqual(b, 1)
            self.assertGreaterEqual(m, 1)
            self.assertEqual(t, b + m)
            rows = load_prepared_examples(out_path)
            self.assertEqual(len(rows), t)
            self.assertTrue(all(hasattr(r, "prompt") and r.prompt for r in rows))

    def test_merge_caps_base_and_mined_counts(self) -> None:
        base_lines = [
            {
                "prompt": f"Question: {i}",
                "response": "<think>x</think><answer>1</answer>",
                "metadata": {"source_id": f"base-{i}"},
            }
            for i in range(3)
        ]
        mined_line = {
            "source_id": "mined-src",
            "problem": "Reach 2 using 1, 1.",
            "raw_trace": (
                "<think>try</think>\n"
                "<answer>1+1</answer>"
            ),
            "recovery_response": (
                "<think>fixed</think>\n<answer>1+1</answer>"
            ),
            "is_correct": False,
            "metadata": {"numbers": [1, 1], "target": 2},
        }
        with tempfile.TemporaryDirectory(prefix="ltr-merge-sft-") as tmp:
            base_path = Path(tmp) / "sft.jsonl"
            mined_path = Path(tmp) / "mined.jsonl"
            out_path = Path(tmp) / "out.jsonl"
            base_path.write_text(
                "\n".join(json.dumps(row) for row in base_lines) + "\n",
                encoding="utf-8",
            )
            mined_path.write_text(json.dumps(mined_line) + "\n", encoding="utf-8")
            b, m, t = merge_sft_train_with_mined_traces(
                sft_train_path=base_path,
                mined_traces_path=mined_path,
                output_path=out_path,
                max_base_examples=2,
                max_mined_examples=1,
            )
            self.assertEqual(b, 2)
            self.assertEqual(m, 1)
            self.assertEqual(t, 3)
            rows = load_prepared_examples(out_path)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0].metadata.get("source_id"), "base-0")
            self.assertEqual(rows[1].metadata.get("source_id"), "base-1")


if __name__ == "__main__":
    unittest.main()

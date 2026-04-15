import json
import tempfile
import unittest
from pathlib import Path

from learning_to_reset.trace_bootstrap import (
    build_bootstrap_negative_record,
    build_bootstrap_trace_corpus,
    write_bootstrap_trace_corpus,
)


class TraceBootstrapTests(unittest.TestCase):
    def test_build_bootstrap_negative_record_converts_trace_to_think_only_negative(self) -> None:
        record = {
            "source_id": "trace-1",
            "problem": "Reach 68.",
            "raw_trace": "<think>Try a subtraction.</think><answer>68</answer>",
            "is_correct": True,
            "metadata": {"source": "positive"},
        }

        negative = build_bootstrap_negative_record(record)

        self.assertEqual(negative["problem"], "Reach 68.")
        self.assertFalse(negative["is_correct"])
        self.assertTrue(negative["source_id"].endswith(":bootstrap-negative"))
        self.assertIn("<think>", negative["raw_trace"])
        self.assertNotIn("<answer>", negative["raw_trace"])

    def test_build_bootstrap_trace_corpus_adds_negative_variants(self) -> None:
        corpus = build_bootstrap_trace_corpus(
            [
                {
                    "source_id": "trace-1",
                    "problem": "Reach 68.",
                    "raw_trace": "<think>Try.</think><answer>68</answer>",
                    "is_correct": True,
                    "metadata": {},
                }
            ]
        )

        self.assertEqual(len(corpus), 2)
        self.assertTrue(corpus[0]["is_correct"])
        self.assertFalse(corpus[1]["is_correct"])

    def test_write_bootstrap_trace_corpus_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "positive.jsonl"
            output_path = Path(tmp_dir) / "bootstrap.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "source_id": "trace-1",
                        "problem": "Reach 68.",
                        "raw_trace": "<think>Try.</think><answer>68</answer>",
                        "is_correct": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = write_bootstrap_trace_corpus(input_path, output_path)
            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(summary["records_written"], 2)
        self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()

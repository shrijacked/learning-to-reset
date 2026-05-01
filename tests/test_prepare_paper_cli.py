from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from pathlib import Path

from learning_to_reset.prepare_paper_artifacts import main


class PreparePaperCliTests(unittest.TestCase):
    def test_main_writes_manifest_for_paper_aligned_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            traces = root / "traces.jsonl"
            countdown_train = root / "countdown-train.jsonl"
            countdown_eval = root / "countdown-eval.jsonl"
            output_dir = root / "prepared"

            traces.write_text(
                json.dumps(
                    {
                        "source_id": "t1",
                        "problem": "Reach 10.",
                        "raw_trace": "<think>Valid.</think><answer>10</answer>",
                        "is_correct": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            countdown_train.write_text(
                json.dumps(
                    {
                        "source_id": "c1",
                        "numbers": [6, 7],
                        "target": 42,
                        "question": "Reach 42 using 6, 7.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            countdown_eval.write_text(
                json.dumps(
                    {
                        "source_id": "c2",
                        "numbers": [9, 8, 3, 1],
                        "target": 24,
                        "question": "Reach 24 using 9, 8, 3, 1.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(
                    [
                        "--traces",
                        str(traces),
                        "--countdown-train",
                        str(countdown_train),
                        "--countdown-eval",
                        str(countdown_eval),
                        "--output-dir",
                        str(output_dir),
                        "--sft-val-ratio",
                        "0.0",
                        "--countdown-val-ratio",
                        "0.0",
                        "--hard-mine-ratio",
                        "0.5",
                    ]
                )

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            hard_eval_exists = (output_dir / "countdown-test-hard.jsonl").exists()
            hard_train_exists = (output_dir / "countdown-train-hard.jsonl").exists()
            hard_mine_exists = (output_dir / "countdown-mine-hard.jsonl").exists()
            hard_eval_raw_exists = (output_dir / "countdown-test-hard-raw.jsonl").exists()
            hard_mine_raw_exists = (output_dir / "countdown-mine-hard-raw.jsonl").exists()

        self.assertEqual(exit_code, 0)
        self.assertEqual(manifest["countdown"]["test"], 1)
        self.assertEqual(manifest["countdown"]["test_hard"], 1)
        self.assertEqual(manifest["countdown"]["train_hard"], 0)
        self.assertEqual(manifest["countdown"]["mine_hard"], 1)
        self.assertTrue(manifest["raw_baseline_variants"])
        self.assertIn('"sft"', buffer.getvalue())
        self.assertTrue(hard_eval_exists)
        self.assertTrue(hard_train_exists)
        self.assertTrue(hard_mine_exists)
        self.assertTrue(hard_eval_raw_exists)
        self.assertTrue(hard_mine_raw_exists)


if __name__ == "__main__":
    unittest.main()

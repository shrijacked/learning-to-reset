from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from pathlib import Path

from learning_to_reset.prepare_artifacts import main


class PrepareCliTests(unittest.TestCase):
    def test_main_writes_manifest_and_split_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            trace_path = base / "traces.json"
            countdown_path = base / "countdown.json"
            output_dir = base / "prepared"

            trace_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "t1",
                            "problem": "Make 10 from 7, 2, 1",
                            "trace": "<think>A</think><answer>10</answer>",
                            "is_correct": True,
                        },
                        {
                            "id": "t2",
                            "problem": "Make 9 from 6, 2, 1",
                            "trace": "<think>B</think><answer>8</answer>",
                            "is_correct": False,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            countdown_path.write_text(
                json.dumps(
                    [
                        {"id": "c1", "numbers": [25, 7, 3, 2], "target": 50},
                        {"id": "c2", "numbers": [9, 8, 3, 1], "target": 24},
                    ]
                ),
                encoding="utf-8",
            )

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(
                    [
                        "--traces",
                        str(trace_path),
                        "--countdown",
                        str(countdown_path),
                        "--output-dir",
                        str(output_dir),
                        "--train-ratio",
                        "0.5",
                        "--val-ratio",
                        "0.0",
                        "--allow-clean-eval",
                    ]
                )

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            countdown_example = json.loads(
                (output_dir / "countdown-train.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue(manifest["allow_clean_eval"])
        self.assertIn("<clean>", countdown_example["prompt"])
        self.assertIn('"sft"', buffer.getvalue())


if __name__ == "__main__":
    unittest.main()

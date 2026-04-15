import json
import tempfile
import unittest
from pathlib import Path

from learning_to_reset.dataset_prep import (
    export_prepared_datasets,
    prepare_countdown_split,
    prepare_trace_split,
)


class DatasetPrepTests(unittest.TestCase):
    def test_prepare_trace_split_curates_examples_and_splits_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "traces.json"
            path.write_text(
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
                        {
                            "id": "t3",
                            "problem": "Make 7 from 5, 2, 1",
                            "trace": "<think>C</think><answer>7</answer>",
                            "is_correct": True,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            split = prepare_trace_split(path, train_ratio=0.67, val_ratio=0.0)

        self.assertEqual(len(split.train), 2)
        self.assertEqual(len(split.validation), 0)
        self.assertEqual(len(split.test), 1)
        self.assertTrue(split.train[1].response.strip().endswith("<clean>"))

    def test_prepare_countdown_split_builds_prompt_only_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "countdown.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "c1", "numbers": [25, 7, 3, 2], "target": 50}),
                        json.dumps({"id": "c2", "numbers": [9, 8, 3, 1], "target": 24}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            split = prepare_countdown_split(path, train_ratio=0.5, val_ratio=0.0, allow_clean=False)

        self.assertEqual(len(split.train), 1)
        self.assertEqual(len(split.test), 1)
        self.assertEqual(split.train[0].response, "")
        self.assertNotIn("<clean>", split.train[0].prompt)

    def test_export_prepared_datasets_writes_jsonl_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            trace_path = base / "traces.json"
            countdown_path = base / "countdown.json"
            out_dir = base / "prepared"

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

            manifest = export_prepared_datasets(
                trace_path=trace_path,
                countdown_path=countdown_path,
                output_dir=out_dir,
                train_ratio=0.5,
                val_ratio=0.0,
            )

            self.assertTrue((out_dir / "manifest.json").exists())
            self.assertTrue((out_dir / "sft-train.jsonl").exists())
            self.assertTrue((out_dir / "countdown-test.jsonl").exists())

            manifest_payload = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            sft_first = json.loads((out_dir / "sft-train.jsonl").read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(manifest["sft"]["train"], 1)
        self.assertEqual(manifest_payload["countdown"]["test"], 1)
        self.assertIn("prompt", sft_first)
        self.assertIn("response", sft_first)
        self.assertIn("metadata", sft_first)


if __name__ == "__main__":
    unittest.main()

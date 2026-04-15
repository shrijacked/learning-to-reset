import json
import tempfile
import unittest
from pathlib import Path

from learning_to_reset.paper_dataset_prep import export_paper_prepared_datasets


class PaperDatasetPrepTests(unittest.TestCase):
    def test_export_paper_prepared_datasets_keeps_eval_split_as_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            traces = root / "traces.jsonl"
            countdown_train = root / "countdown-train.jsonl"
            countdown_eval = root / "countdown-eval.jsonl"
            output_dir = root / "prepared"

            traces.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "source_id": "t1",
                                "problem": "Reach 10.",
                                "raw_trace": "<think>Valid.</think><answer>10</answer>",
                                "is_correct": True,
                            }
                        ),
                        json.dumps(
                            {
                                "source_id": "t2",
                                "problem": "Reach 9.",
                                "raw_trace": "<think>Wrong.</think><answer>8</answer>",
                                "is_correct": False,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            countdown_train.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "source_id": "c1",
                                "numbers": [25, 7, 3, 2],
                                "target": 50,
                                "question": "Reach 50 using 25, 7, 3, 2.",
                            }
                        ),
                        json.dumps(
                            {
                                "source_id": "c2",
                                "numbers": [9, 8, 3, 1],
                                "target": 24,
                                "question": "Reach 24 using 9, 8, 3, 1.",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            countdown_eval.write_text(
                json.dumps(
                    {
                        "source_id": "c3",
                        "numbers": [60, 27, 19],
                        "target": 68,
                        "question": "Reach 68 using 60, 27, 19.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = export_paper_prepared_datasets(
                trace_path=traces,
                countdown_train_path=countdown_train,
                countdown_eval_path=countdown_eval,
                output_dir=output_dir,
                sft_val_ratio=0.5,
                countdown_val_ratio=0.5,
                allow_clean=True,
            )
            test_payload = (output_dir / "countdown-test.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(manifest["countdown"]["test"], 1)
        self.assertEqual(len(test_payload), 1)
        self.assertIn('"source_id": "c3"', test_payload[0])

    def test_export_paper_prepared_datasets_can_include_retry_recovery_examples(self) -> None:
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
                        "raw_trace": "<think>Wrong.</think><answer>9</answer>",
                        "is_correct": False,
                        "recovery_response": "<think>Fresh.</think><answer>10</answer>",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            countdown_train.write_text(
                json.dumps(
                    {
                        "source_id": "c1",
                        "numbers": [25, 7, 3, 2],
                        "target": 50,
                        "question": "Reach 50 using 25, 7, 3, 2.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            countdown_eval.write_text(
                json.dumps(
                    {
                        "source_id": "c2",
                        "numbers": [60, 27, 19],
                        "target": 68,
                        "question": "Reach 68 using 60, 27, 19.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = export_paper_prepared_datasets(
                trace_path=traces,
                countdown_train_path=countdown_train,
                countdown_eval_path=countdown_eval,
                output_dir=output_dir,
                sft_val_ratio=0.0,
                countdown_val_ratio=0.0,
                allow_clean=True,
                include_recovery_examples=True,
                recovery_repeat=2,
            )
            sft_train_lines = (output_dir / "sft-train.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(manifest["sft"]["train"], 3)
        self.assertEqual(len(sft_train_lines), 3)
        self.assertIn('"stage": "retry-recovery"', sft_train_lines[1])


if __name__ == "__main__":
    unittest.main()

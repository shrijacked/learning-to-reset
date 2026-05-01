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
                hard_mine_ratio=0.5,
                allow_clean=True,
            )
            test_payload = (output_dir / "countdown-test.jsonl").read_text(encoding="utf-8").splitlines()
            raw_test_payload = (
                output_dir / "countdown-test-raw.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            sft_mix_summary = json.loads(
                (output_dir / "sft-mix-summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["countdown"]["test"], 1)
        self.assertEqual(len(test_payload), 1)
        self.assertIn('"source_id": "c3"', test_payload[0])
        self.assertEqual(manifest["countdown"]["test_hard"], 0)
        self.assertEqual(len(raw_test_payload), 1)
        self.assertIn('"source_id": "c3"', raw_test_payload[0])
        self.assertNotIn("emit <clean>", raw_test_payload[0])
        self.assertIn("before_filter", sft_mix_summary)
        self.assertIn("after_filter", sft_mix_summary)

    def test_export_paper_prepared_datasets_writes_hard_eval_slice(self) -> None:
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
                        "source_id": "train",
                        "numbers": [1, 2],
                        "target": 3,
                        "question": "Reach 3 using 1, 2.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            countdown_eval.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "source_id": "easy",
                                "numbers": [20, 5, 99],
                                "target": 25,
                                "question": "Reach 25 using 20, 5, 99.",
                            }
                        ),
                        json.dumps(
                            {
                                "source_id": "hard",
                                "numbers": [6, 7],
                                "target": 42,
                                "question": "Reach 42 using 6, 7.",
                            }
                        ),
                    ]
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
                hard_mine_ratio=0.5,
                allow_clean=True,
            )
            hard_payload = (output_dir / "countdown-test-hard.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(manifest["countdown"]["test"], 2)
        self.assertEqual(manifest["countdown"]["test_hard"], 1)
        self.assertEqual(len(hard_payload), 1)
        self.assertIn('"source_id": "hard"', hard_payload[0])

    def test_export_paper_prepared_datasets_writes_disjoint_hard_train_and_mine_slices(self) -> None:
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
                "\n".join(
                    [
                        json.dumps(
                            {
                                "source_id": "train-hard-1",
                                "numbers": [6, 7],
                                "target": 42,
                                "question": "Reach 42 using 6, 7.",
                            }
                        ),
                        json.dumps(
                            {
                                "source_id": "train-hard-2",
                                "numbers": [8, 9],
                                "target": 72,
                                "question": "Reach 72 using 8, 9.",
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
                        "source_id": "eval-hard",
                        "numbers": [4, 11],
                        "target": 44,
                        "question": "Reach 44 using 4, 11.",
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
                hard_mine_ratio=0.5,
                allow_clean=True,
            )
            hard_train_payload = (
                output_dir / "countdown-train-hard.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            hard_mine_payload = (
                output_dir / "countdown-mine-hard.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            hard_mine_raw_payload = (
                output_dir / "countdown-mine-hard-raw.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            hard_holdout_payload = (
                output_dir / "countdown-test-hard.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            hard_holdout_raw_payload = (
                output_dir / "countdown-test-hard-raw.jsonl"
            ).read_text(encoding="utf-8").splitlines()

        self.assertEqual(manifest["countdown"]["train_hard"], 1)
        self.assertEqual(manifest["countdown"]["mine_hard"], 1)
        self.assertEqual(manifest["countdown"]["test_hard"], 1)
        self.assertEqual(len(hard_train_payload), 1)
        self.assertEqual(len(hard_mine_payload), 1)
        self.assertEqual(len(hard_mine_raw_payload), 1)
        self.assertEqual(len(hard_holdout_payload), 1)
        self.assertEqual(len(hard_holdout_raw_payload), 1)
        self.assertIn('"source_id": "train-hard-1"', hard_train_payload[0])
        self.assertIn('"source_id": "train-hard-2"', hard_mine_payload[0])
        self.assertIn('"source_id": "eval-hard"', hard_holdout_payload[0])
        self.assertNotIn("emit <clean>", hard_mine_raw_payload[0])
        self.assertNotIn("emit <clean>", hard_holdout_raw_payload[0])

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
                hard_mine_ratio=0.2,
                allow_clean=True,
                include_recovery_examples=True,
                recovery_repeat=2,
            )
            sft_train_lines = (output_dir / "sft-train.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(manifest["sft"]["train"], 3)
        self.assertEqual(len(sft_train_lines), 3)
        self.assertIn('"stage": "retry-recovery"', sft_train_lines[1])

    def test_export_paper_prepared_datasets_can_filter_retry_recovery_examples(self) -> None:
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
                                "source_id": "good",
                                "problem": "Reach 10 using 7, 2, 1.",
                                "raw_trace": "<think>Wrong.</think><answer>9</answer>",
                                "is_correct": False,
                                "numbers": [7, 2, 1],
                                "target": 10,
                                "recovery_response": (
                                    "<think>Fresh.</think><answer>((7 + 2) + 1)</answer>"
                                ),
                            }
                        ),
                        json.dumps(
                            {
                                "source_id": "bad",
                                "problem": "Reach 10 using 7, 2, 1.",
                                "raw_trace": "<think>Wrong.</think><answer>9</answer>",
                                "is_correct": False,
                                "numbers": [7, 2, 1],
                                "target": 10,
                                "recovery_response": (
                                    "<think>Still wrong.</think><answer>(7 + 2)</answer>"
                                ),
                            }
                        ),
                    ]
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
                hard_mine_ratio=0.2,
                allow_clean=True,
                include_recovery_examples=True,
                require_recovery_target_correct=True,
            )
            sft_train_lines = (output_dir / "sft-train.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(manifest["sft"]["train"], 3)
        self.assertTrue(manifest["require_recovery_target_correct"])
        self.assertEqual(sum('"stage": "retry-recovery"' in line for line in sft_train_lines), 1)
        self.assertTrue(any('"source_id": "good"' in line for line in sft_train_lines))

    def test_export_paper_prepared_datasets_can_exclude_bootstrap_negatives(self) -> None:
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
                                "source_id": "positive",
                                "problem": "Reach 10.",
                                "raw_trace": "<think>Valid.</think><answer>10</answer>",
                                "is_correct": True,
                                "metadata": {"source": "reference"},
                            }
                        ),
                        json.dumps(
                            {
                                "source_id": "bootstrap-negative",
                                "problem": "Reach 10.",
                                "raw_trace": "<think>Confused.</think>",
                                "is_correct": False,
                                "metadata": {
                                    "source": "reference",
                                    "bootstrap_kind": "think_only_negative",
                                },
                            }
                        ),
                    ]
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

            manifest = export_paper_prepared_datasets(
                trace_path=traces,
                countdown_train_path=countdown_train,
                countdown_eval_path=countdown_eval,
                output_dir=output_dir,
                sft_val_ratio=0.0,
                countdown_val_ratio=0.0,
                allow_clean=True,
                exclude_bootstrap_negatives=True,
            )
            sft_train_lines = (output_dir / "sft-train.jsonl").read_text(encoding="utf-8").splitlines()
            sft_mix_summary = json.loads(
                (output_dir / "sft-mix-summary.json").read_text(encoding="utf-8")
            )

        self.assertTrue(manifest["exclude_bootstrap_negatives"])
        self.assertEqual(manifest["sft"]["train"], 1)
        self.assertEqual(len(sft_train_lines), 1)
        self.assertIn('"source_id": "positive"', sft_train_lines[0])
        self.assertEqual(
            sft_mix_summary["before_filter"]["by_bootstrap_kind"]["think_only_negative"],
            1,
        )
        self.assertEqual(
            sft_mix_summary["after_filter"].get("by_bootstrap_kind", {}),
            {},
        )


if __name__ == "__main__":
    unittest.main()

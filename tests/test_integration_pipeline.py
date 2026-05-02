import json
import tempfile
import unittest
from pathlib import Path

from learning_to_reset.eval_runtime import evaluate_countdown_outputs, run_multi_clean_eval_loop
from learning_to_reset.failure_recovery_traces import generate_failure_recovery_trace_corpus
from learning_to_reset.paper_dataset_prep import export_paper_prepared_datasets
from learning_to_reset.sft_runtime import load_prepared_examples


class TinyPipelineIntegrationTests(unittest.TestCase):
    def test_full_cpu_tiny_eval_mine_reprep_retry_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            traces = root / "traces.jsonl"
            countdown_train = root / "countdown-train.jsonl"
            countdown_eval = root / "countdown-eval.jsonl"
            first_artifacts = root / "artifacts-1"
            mined_recoveries = root / "mined-recoveries.jsonl"
            combined_traces = root / "combined-traces.jsonl"
            second_artifacts = root / "artifacts-2"

            traces.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "source_id": "trace-positive",
                                "problem": "Reach 24 using 9, 8, 3, 1.",
                                "raw_trace": "<think>Use 8 and 3.</think><answer>8 * 3</answer>",
                                "is_correct": True,
                                "metadata": {"source": "reference"},
                            }
                        ),
                        json.dumps(
                            {
                                "source_id": "trace-negative",
                                "problem": "Reach 24 using 9, 8, 3, 1.",
                                "raw_trace": "<think>Bad path.</think><answer>9 + 8</answer>",
                                "is_correct": False,
                                "metadata": {"source": "reference"},
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
                        "source_id": "train-hard",
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
                        "source_id": "eval-hard",
                        "numbers": [8, 3],
                        "target": 24,
                        "question": "Reach 24 using 8, 3.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            export_paper_prepared_datasets(
                trace_path=traces,
                countdown_train_path=countdown_train,
                countdown_eval_path=countdown_eval,
                output_dir=first_artifacts,
                sft_val_ratio=0.0,
                countdown_val_ratio=0.0,
                hard_mine_ratio=0.0,
            )
            hard_holdout = load_prepared_examples(first_artifacts / "countdown-test-hard.jsonl")
            raw_summary = evaluate_countdown_outputs(
                hard_holdout,
                ["<think>I should reset before answering.</think><clean>"],
            )
            raw_results = root / "raw-results.jsonl"
            raw_results.write_text(
                "\n".join(json.dumps(row) for row in raw_summary["results"]) + "\n",
                encoding="utf-8",
            )

            mining_summary = generate_failure_recovery_trace_corpus(
                prepared_countdown_path=first_artifacts / "countdown-test-hard.jsonl",
                eval_results_path=raw_results,
                output_path=mined_recoveries,
                recovery_style="verification",
            )
            combined_traces.write_text(
                traces.read_text(encoding="utf-8")
                + mined_recoveries.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            second_manifest = export_paper_prepared_datasets(
                trace_path=combined_traces,
                countdown_train_path=countdown_train,
                countdown_eval_path=countdown_eval,
                output_dir=second_artifacts,
                sft_val_ratio=0.0,
                countdown_val_ratio=0.0,
                hard_mine_ratio=0.0,
                include_recovery_examples=True,
                require_recovery_target_correct=True,
            )

            calls_by_prompt = {}

            def fake_generator(prompt: str, allow_clean: bool) -> str:
                calls_by_prompt[prompt] = calls_by_prompt.get(prompt, 0) + 1
                if allow_clean:
                    return "<think>This attempt is tangled.</think><clean>"
                return "<think>Fresh pass.</think><answer>8 * 3</answer>"

            retry_summary = run_multi_clean_eval_loop(
                load_prepared_examples(second_artifacts / "countdown-test-hard.jsonl"),
                generator=fake_generator,
                max_clean_tries=1,
            )

        self.assertEqual(raw_summary["valid"], 0)
        self.assertEqual(mining_summary["records_written"], 1)
        self.assertEqual(second_manifest["sft"]["provenance"]["after_filter"]["total_examples"], 4)
        self.assertEqual(retry_summary["valid"], 1)
        self.assertEqual(retry_summary["correct"], 1)
        self.assertEqual(retry_summary["clean_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

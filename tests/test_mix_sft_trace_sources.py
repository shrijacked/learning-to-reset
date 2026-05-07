import json
import tempfile
import unittest
from pathlib import Path

from learning_to_reset.mix_sft_trace_sources import build_mixed_trace_source


class MixSftTraceSourcesTests(unittest.TestCase):
    def test_build_mixed_trace_source_samples_reference_deterministically(self) -> None:
        synthetic_rows = [
            {
                "source_id": f"syn-{index}",
                "problem": "Reach 2.",
                "raw_trace": "<think>x</think><answer>2</answer>",
                "is_correct": True,
            }
            for index in range(9)
        ]
        reference_rows = [
            {
                "source_id": f"ref-{index}",
                "problem": "Reference task.",
                "raw_trace": "<think>x</think><answer>2</answer>",
                "is_correct": True,
            }
            for index in range(5)
        ]
        with tempfile.TemporaryDirectory(prefix="ltr-mix-sft-") as tmp:
            root = Path(tmp)
            synthetic_path = root / "synthetic.jsonl"
            reference_path = root / "reference.jsonl"
            output_a = root / "mixed-a.jsonl"
            output_b = root / "mixed-b.jsonl"
            synthetic_path.write_text(
                "\n".join(json.dumps(row) for row in synthetic_rows) + "\n",
                encoding="utf-8",
            )
            reference_path.write_text(
                "\n".join(json.dumps(row) for row in reference_rows) + "\n",
                encoding="utf-8",
            )

            summary_a = build_mixed_trace_source(
                synthetic_path=synthetic_path,
                reference_path=reference_path,
                output_path=output_a,
                reference_ratio=0.1,
                seed=7,
            )
            summary_b = build_mixed_trace_source(
                synthetic_path=synthetic_path,
                reference_path=reference_path,
                output_path=output_b,
                reference_ratio=0.1,
                seed=7,
            )
            rows = [
                json.loads(line)
                for line in output_a.read_text(encoding="utf-8").splitlines()
            ]
            rendered_a = output_a.read_text(encoding="utf-8")
            rendered_b = output_b.read_text(encoding="utf-8")

        self.assertEqual(summary_a["reference_rows_sampled"], 1)
        summary_a_without_path = dict(summary_a)
        summary_b_without_path = dict(summary_b)
        summary_a_without_path.pop("output_path")
        summary_b_without_path.pop("output_path")
        self.assertEqual(summary_a_without_path, summary_b_without_path)
        self.assertEqual(rendered_a, rendered_b)
        self.assertEqual(len(rows), 10)
        domains = [row["metadata"]["trace_domain"] for row in rows]
        self.assertEqual(domains.count("countdown-synthetic"), 9)
        self.assertEqual(domains.count("reference-behavior"), 1)

    def test_build_mixed_trace_source_rejects_invalid_ratio(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ltr-mix-sft-") as tmp:
            root = Path(tmp)
            synthetic_path = root / "synthetic.jsonl"
            reference_path = root / "reference.jsonl"
            synthetic_path.write_text("", encoding="utf-8")
            reference_path.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "reference_ratio"):
                build_mixed_trace_source(
                    synthetic_path=synthetic_path,
                    reference_path=reference_path,
                    output_path=root / "mixed.jsonl",
                    reference_ratio=1.1,
                )


if __name__ == "__main__":
    unittest.main()

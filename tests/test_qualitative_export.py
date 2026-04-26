import json
import tempfile
import unittest
from pathlib import Path

from learning_to_reset.qualitative_export import (
    render_qualitative_markdown,
    select_qualitative_samples,
    write_qualitative_export,
)


def _record(
    *,
    source_id: str,
    is_valid: bool,
    reaches_target: bool,
    cleaned: bool,
    score: float,
    response: str = "",
    reason: str = "",
) -> dict:
    return {
        "source_id": source_id,
        "prompt": f"Question: {source_id}.",
        "response": response or f"<think>{source_id}</think><answer>1+1</answer>",
        "expression": "1+1",
        "value": "2",
        "is_valid": is_valid,
        "reaches_target": reaches_target,
        "reason": reason,
        "score": score,
        "cleaned": cleaned,
    }


class QualitativeExportTests(unittest.TestCase):
    def test_select_returns_best_correct_cleaned_and_worst_invalid_no_clean(self) -> None:
        records = [
            _record(source_id="good-1", is_valid=True, reaches_target=True, cleaned=True, score=1.1),
            _record(source_id="good-2", is_valid=True, reaches_target=True, cleaned=True, score=1.05),
            _record(source_id="meh-1", is_valid=True, reaches_target=False, cleaned=True, score=0.1),
            _record(source_id="bad-1", is_valid=False, reaches_target=False, cleaned=False, score=0.0),
            _record(source_id="bad-2", is_valid=False, reaches_target=False, cleaned=False, score=0.0),
            _record(source_id="bad-3", is_valid=False, reaches_target=False, cleaned=False, score=0.0),
        ]

        selection = select_qualitative_samples(records, n_best=2, n_worst=2)

        self.assertEqual([row["source_id"] for row in selection["best"]], ["good-1", "good-2"])
        self.assertEqual(
            [row["source_id"] for row in selection["worst"]],
            ["bad-1", "bad-2"],
        )

    def test_select_handles_fewer_records_than_requested(self) -> None:
        records = [
            _record(source_id="good", is_valid=True, reaches_target=True, cleaned=True, score=1.1),
        ]

        selection = select_qualitative_samples(records, n_best=3, n_worst=3)

        self.assertEqual(len(selection["best"]), 1)
        self.assertEqual(selection["worst"], [])

    def test_render_markdown_contains_section_headers_and_records(self) -> None:
        selection = {
            "best": [
                _record(source_id="good", is_valid=True, reaches_target=True, cleaned=True, score=1.1),
            ],
            "worst": [
                _record(source_id="bad", is_valid=False, reaches_target=False, cleaned=False, score=0.0),
            ],
        }

        markdown = render_qualitative_markdown(selection, n_best=1, n_worst=1)

        self.assertIn("# Qualitative samples", markdown)
        self.assertIn("## Best (target-correct, used reset)", markdown)
        self.assertIn("## Worst (invalid, no reset)", markdown)
        self.assertIn("good", markdown)
        self.assertIn("bad", markdown)

    def test_write_qualitative_export_creates_markdown_file_from_results(self) -> None:
        records = [
            _record(source_id="good", is_valid=True, reaches_target=True, cleaned=True, score=1.1),
            _record(source_id="bad", is_valid=False, reaches_target=False, cleaned=False, score=0.0),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            results_path = root / "results.jsonl"
            results_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            output_path = root / "qualitative.md"

            write_qualitative_export(
                results_path=results_path,
                output_path=output_path,
                n_best=1,
                n_worst=1,
            )

            text = output_path.read_text(encoding="utf-8")

        self.assertIn("good", text)
        self.assertIn("bad", text)


if __name__ == "__main__":
    unittest.main()

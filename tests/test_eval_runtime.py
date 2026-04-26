import unittest

from learning_to_reset.eval_runtime import (
    build_countdown_sample_from_example,
    evaluate_countdown_outputs,
)
from learning_to_reset.prompts import PromptExample


class EvalRuntimeTests(unittest.TestCase):
    def test_build_countdown_sample_from_example_uses_metadata(self) -> None:
        example = PromptExample(
            prompt="Question: reach 50",
            response="",
            metadata={
                "numbers": (25, 7, 3, 2),
                "target": 50,
                "source_id": "c1",
                "question": "Reach 50 using 25, 7, 3, 2.",
            },
        )

        sample = build_countdown_sample_from_example(example)

        self.assertEqual(sample.numbers, (25, 7, 3, 2))
        self.assertEqual(sample.target, 50)
        self.assertEqual(sample.source_id, "c1")
        self.assertEqual(sample.question, "Reach 50 using 25, 7, 3, 2.")

    def test_evaluate_countdown_outputs_computes_summary(self) -> None:
        examples = (
            PromptExample(
                prompt="Question: reach 50",
                response="",
                metadata={"numbers": (25, 7, 3, 2), "target": 50, "source_id": "c1"},
            ),
            PromptExample(
                prompt="Question: reach 24",
                response="",
                metadata={"numbers": (9, 8, 3, 1), "target": 24, "source_id": "c2"},
            ),
        )
        responses = (
            "<think>Try.</think><answer>25 * 2</answer>",
            "<think>Wrong.</think><answer>9 + 8 + 3 + 1</answer>",
        )

        summary = evaluate_countdown_outputs(examples, responses)

        self.assertEqual(summary["total_examples"], 2)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["valid"], 2)
        self.assertAlmostEqual(summary["accuracy"], 0.5)
        self.assertAlmostEqual(summary["valid_rate"], 1.0)
        self.assertAlmostEqual(summary["average_score"], 0.6)
        self.assertEqual(summary["results"][0]["value"], "50")
        self.assertEqual(summary["results"][1]["value"], "21")

    def test_evaluate_countdown_outputs_reports_clean_rate_and_score_when_cleaned(self) -> None:
        examples = (
            PromptExample(
                prompt="Question: reach 50",
                response="",
                metadata={"numbers": (25, 7, 3, 2), "target": 50, "source_id": "c1"},
            ),
            PromptExample(
                prompt="Question: reach 24",
                response="",
                metadata={"numbers": (9, 8, 3, 1), "target": 24, "source_id": "c2"},
            ),
            PromptExample(
                prompt="Question: reach 12",
                response="",
                metadata={"numbers": (6, 4, 2, 1), "target": 12, "source_id": "c3"},
            ),
        )
        responses = (
            "<think>Try.</think><clean>\n<think>Retry.</think><answer>25 * 2</answer>",
            "<think>Wrong.</think><answer>9 + 8 + 3 + 1</answer>",
            "<think>Bad path.</think><clean>\n<think>Try.</think><answer>nonsense</answer>",
        )

        summary = evaluate_countdown_outputs(examples, responses)

        self.assertIn("clean_rate", summary)
        self.assertIn("score_when_cleaned", summary)
        self.assertAlmostEqual(summary["clean_rate"], 2 / 3)
        self.assertAlmostEqual(summary["score_when_cleaned"], (1.1 + 0.0) / 2)
        self.assertTrue(summary["results"][0]["cleaned"])
        self.assertFalse(summary["results"][1]["cleaned"])
        self.assertTrue(summary["results"][2]["cleaned"])

    def test_evaluate_countdown_outputs_score_when_cleaned_is_zero_with_no_cleans(self) -> None:
        examples = (
            PromptExample(
                prompt="Question: reach 24",
                response="",
                metadata={"numbers": (9, 8, 3, 1), "target": 24, "source_id": "c1"},
            ),
        )
        responses = ("<think>No clean.</think><answer>9 + 8 + 3 + 1</answer>",)

        summary = evaluate_countdown_outputs(examples, responses)

        self.assertAlmostEqual(summary["clean_rate"], 0.0)
        self.assertAlmostEqual(summary["score_when_cleaned"], 0.0)


if __name__ == "__main__":
    unittest.main()

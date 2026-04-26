import unittest

from learning_to_reset.countdown_verifier import VerificationResult
from learning_to_reset.data import CountdownSample
from learning_to_reset.eval_runtime import (
    _build_retry_prompt,
    build_countdown_sample_from_example,
    evaluate_countdown_outputs,
    format_verifier_feedback,
    run_multi_clean_eval_loop,
)
from learning_to_reset.prompts import PromptExample, build_reasoning_prompt


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


def _build_reach_50_example() -> PromptExample:
    question = "Reach 50 using 25, 7, 3, 2."
    return PromptExample(
        prompt=build_reasoning_prompt(question, allow_clean=True),
        response="",
        metadata={
            "numbers": (25, 7, 3, 2),
            "target": 50,
            "source_id": "c1",
            "question": question,
        },
    )


class MultiCleanEvalLoopTests(unittest.TestCase):
    def test_default_max_clean_tries_one_matches_legacy_behaviour(self) -> None:
        example = _build_reach_50_example()
        scripted = iter([
            "<think>Confused.</think><clean>",
            "<think>Fresh.</think><answer>25 * 2</answer>",
        ])

        def generator(prompt: str, allow_clean: bool) -> str:
            return next(scripted)

        summary = run_multi_clean_eval_loop(
            (example,), generator=generator, max_clean_tries=1
        )

        self.assertEqual(summary["total_examples"], 1)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["results"][0]["clean_count"], 1)
        self.assertEqual(summary["results"][0]["segment_count"], 2)
        self.assertTrue(summary["results"][0]["cleaned"])

    def test_multi_clean_keeps_retrying_until_target_correct(self) -> None:
        example = _build_reach_50_example()
        scripted = iter([
            "<think>Bad.</think><clean>",
            "<think>Still bad.</think><clean>",
            "<think>Got it.</think><answer>25 * 2</answer>",
        ])

        def generator(prompt: str, allow_clean: bool) -> str:
            return next(scripted)

        summary = run_multi_clean_eval_loop(
            (example,), generator=generator, max_clean_tries=3
        )

        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["results"][0]["clean_count"], 2)
        self.assertEqual(summary["results"][0]["segment_count"], 3)
        self.assertTrue(summary["results"][0]["budget_exhausted"] is False)

    def test_multi_clean_budget_caps_retries_when_no_correct_answer(self) -> None:
        example = _build_reach_50_example()
        scripted = iter([
            "<think>Bad.</think><clean>",
            "<think>Bad.</think><clean>",
            "<think>Bad.</think><clean>",
            "<think>Last.</think><answer>1 + 2</answer>",
        ])
        called_with_allow_clean = []

        def generator(prompt: str, allow_clean: bool) -> str:
            called_with_allow_clean.append(allow_clean)
            return next(scripted)

        summary = run_multi_clean_eval_loop(
            (example,), generator=generator, max_clean_tries=3
        )

        self.assertEqual(summary["correct"], 0)
        self.assertEqual(summary["results"][0]["clean_count"], 3)
        self.assertEqual(summary["results"][0]["segment_count"], 4)
        self.assertTrue(summary["results"][0]["budget_exhausted"])
        self.assertEqual(called_with_allow_clean[-1], False)

    def test_multi_clean_short_circuits_on_first_correct_answer(self) -> None:
        example = _build_reach_50_example()
        scripted = iter([
            "<think>Direct hit.</think><answer>25 * 2</answer>",
            "<think>Should not run.</think><answer>nonsense</answer>",
        ])

        def generator(prompt: str, allow_clean: bool) -> str:
            return next(scripted)

        summary = run_multi_clean_eval_loop(
            (example,), generator=generator, max_clean_tries=3
        )

        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["results"][0]["clean_count"], 0)
        self.assertEqual(summary["results"][0]["segment_count"], 1)
        self.assertFalse(summary["results"][0]["cleaned"])

    def test_multi_clean_rejects_invalid_max_clean_tries(self) -> None:
        example = _build_reach_50_example()

        def generator(prompt: str, allow_clean: bool) -> str:
            return "<answer>25 * 2</answer>"

        with self.assertRaises(ValueError):
            run_multi_clean_eval_loop(
                (example,), generator=generator, max_clean_tries=0
            )

    def test_verifier_feedback_appends_truth_to_retry_prompt(self) -> None:
        example = _build_reach_50_example()
        prompts: list[str] = []
        scripted = iter(
            [
                "<t></t><answer>25 + 7</answer><clean>",
                "<t></t><answer>25 * 2</answer>",
            ]
        )

        def generator(prompt: str, allow_clean: bool) -> str:
            prompts.append(prompt)
            return next(scripted)

        summary = run_multi_clean_eval_loop(
            (example,),
            generator=generator,
            max_clean_tries=1,
            verifier_feedback=True,
        )

        self.assertEqual(len(prompts), 2)
        self.assertIn("Verifier feedback", prompts[1])
        self.assertIn("evaluates to 32", prompts[1])
        self.assertIn("50", prompts[1])
        self.assertEqual(summary["correct"], 1)
        self.assertTrue(summary["verifier_feedback"])

    def test_verifier_feedback_false_preserves_plain_retry_prompt(self) -> None:
        example = _build_reach_50_example()
        base = _build_retry_prompt(example)
        prompts: list[str] = []
        scripted = iter(
            [
                "<t></t><answer>25 + 7</answer><clean>",
                "<t></t><answer>25 * 2</answer>",
            ]
        )

        def generator(prompt: str, allow_clean: bool) -> str:
            prompts.append(prompt)
            return next(scripted)

        run_multi_clean_eval_loop(
            (example,),
            generator=generator,
            max_clean_tries=1,
            verifier_feedback=False,
        )

        self.assertEqual(prompts[1], base)


class VerifierFeedbackFormattingTests(unittest.TestCase):
    def test_invalid_reason_wired_into_feedback_text(self) -> None:
        sample = CountdownSample(
            numbers=(2, 3),
            target=6,
            question="q",
        )
        vr = VerificationResult(
            expression=None,
            is_valid=False,
            reaches_target=False,
            used_numbers=(),
            value=None,
            reason="No <answer> block found in response.",
        )
        text = format_verifier_feedback(vr, sample)
        self.assertIn("No <answer> block", text)
        self.assertIn("6", text)
        self.assertIn("2, 3", text)


if __name__ == "__main__":
    unittest.main()

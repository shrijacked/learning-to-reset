import unittest

from learning_to_reset.countdown_verifier import (
    extract_answer_expression,
    score_countdown_response,
    verify_countdown_expression,
)
from learning_to_reset.data import CountdownSample


class CountdownVerifierTests(unittest.TestCase):
    def test_extract_answer_expression_returns_last_answer_block(self) -> None:
        response = "<answer>bad</answer><think>retry</think><answer>(60 + 27) - 19</answer>"

        expression = extract_answer_expression(response)

        self.assertEqual(expression, "(60 + 27) - 19")

    def test_verify_countdown_expression_accepts_valid_solution(self) -> None:
        result = verify_countdown_expression("(60 + 27) - 19", numbers=(60, 27, 19), target=68)

        self.assertTrue(result.is_valid)
        self.assertTrue(result.reaches_target)
        self.assertEqual(result.used_numbers, (60, 27, 19))

    def test_verify_countdown_expression_rejects_number_reuse(self) -> None:
        result = verify_countdown_expression("25 + 25", numbers=(25, 7, 3, 2), target=50)

        self.assertFalse(result.is_valid)
        self.assertIn("unused or repeated", result.reason)

    def test_verify_countdown_expression_rejects_wrong_target(self) -> None:
        result = verify_countdown_expression("25 + 7 + 3 + 2", numbers=(25, 7, 3, 2), target=50)

        self.assertTrue(result.is_valid)
        self.assertFalse(result.reaches_target)

    def test_score_countdown_response_flags_missing_answer(self) -> None:
        sample = CountdownSample(
            source_id="sample-1",
            numbers=(25, 7, 3, 2),
            target=50,
            question="Use the numbers 25, 7, 3, 2 to reach 50.",
        )

        result = score_countdown_response("<think>No answer yet.</think>", sample)

        self.assertFalse(result.is_valid)
        self.assertIn("No <answer>", result.reason)


if __name__ == "__main__":
    unittest.main()

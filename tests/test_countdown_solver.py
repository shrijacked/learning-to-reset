import unittest

from learning_to_reset.countdown_solver import solve_countdown, solve_countdown_variants
from learning_to_reset.countdown_verifier import verify_countdown_expression


class CountdownSolverTests(unittest.TestCase):
    def test_solve_countdown_finds_valid_expression(self) -> None:
        expression = solve_countdown((9, 11, 12, 17), 70)

        self.assertIsNotNone(expression)
        verification = verify_countdown_expression(
            expression,
            numbers=(9, 11, 12, 17),
            target=70,
        )
        self.assertTrue(verification.is_valid)
        self.assertTrue(verification.reaches_target)

    def test_solve_countdown_returns_none_when_no_solution_exists(self) -> None:
        expression = solve_countdown((2, 2), 5)

        self.assertIsNone(expression)

    def test_solve_countdown_variants_returns_multiple_distinct_valid_solutions(self) -> None:
        expressions = solve_countdown_variants((9, 11, 12, 17), 70, max_solutions=3)

        self.assertGreaterEqual(len(expressions), 2)
        self.assertEqual(len(expressions), len(set(expressions)))
        for expression in expressions:
            verification = verify_countdown_expression(
                expression,
                numbers=(9, 11, 12, 17),
                target=70,
            )
            self.assertTrue(verification.is_valid)
            self.assertTrue(verification.reaches_target)


if __name__ == "__main__":
    unittest.main()

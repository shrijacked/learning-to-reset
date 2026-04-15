import unittest

from learning_to_reset.countdown_solver import solve_countdown
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


if __name__ == "__main__":
    unittest.main()


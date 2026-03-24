import unittest

from learning_to_reset.context_manager import (
    build_initial_prompt,
    build_retry_prompt,
    manage_single_clean_cycle,
)


class ContextManagerTests(unittest.TestCase):
    def test_returns_initial_answer_when_no_clean_is_emitted(self) -> None:
        managed = manage_single_clean_cycle(
            question="Make 68 from 60, 27, 19",
            base_instructions="Solve carefully.",
            clean_instructions="Emit <clean> if needed.",
            initial_response="<think>Useful reasoning.</think><answer>68</answer>",
        )

        self.assertFalse(managed.cleaned)
        self.assertIsNone(managed.retry_prompt)
        self.assertEqual(managed.final_response, managed.initial_response)
        self.assertEqual(managed.final_answer, "68")

    def test_clean_emission_requires_retry_response(self) -> None:
        with self.assertRaises(ValueError):
            manage_single_clean_cycle(
                question="Make 55 from 61, 63, 57",
                base_instructions="Solve carefully.",
                clean_instructions="Emit <clean> if needed.",
                initial_response="<think>Confusing.</think><clean>",
            )

    def test_clean_emission_builds_fresh_retry_prompt_and_uses_retry_answer(self) -> None:
        managed = manage_single_clean_cycle(
            question="Make 55 from 61, 63, 57",
            base_instructions="Solve carefully.",
            clean_instructions="Emit <clean> if needed.",
            initial_response="<think>Confusing.</think><clean>",
            retry_response="<think>Fresh start.</think><answer>55</answer>",
        )

        self.assertTrue(managed.cleaned)
        self.assertEqual(
            managed.retry_prompt,
            build_retry_prompt("Make 55 from 61, 63, 57", "Solve carefully."),
        )
        self.assertEqual(managed.final_response, "<think>Fresh start.</think><answer>55</answer>")
        self.assertEqual(managed.final_answer, "55")

    def test_retry_prompt_removes_clean_instructions(self) -> None:
        initial_prompt = build_initial_prompt(
            question="Q",
            base_instructions="Base",
            clean_instructions="Clean instructions",
        )
        retry_prompt = build_retry_prompt(question="Q", base_instructions="Base")

        self.assertIn("Clean instructions", initial_prompt)
        self.assertNotIn("Clean instructions", retry_prompt)


if __name__ == "__main__":
    unittest.main()

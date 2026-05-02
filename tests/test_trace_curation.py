import unittest

from learning_to_reset.trace_curation import curate_trace, normalize_trace


BASE_INSTRUCTIONS = "Solve the problem carefully."
CLEAN_INSTRUCTIONS = "If your search becomes confusing, emit <clean>."


class TraceCurationTests(unittest.TestCase):
    def test_normalize_trace_merges_think_blocks_and_keeps_final_answer(self) -> None:
        raw_trace = """
        ignored prefix
        <think>
        First attempt.
        </think>
        ignored middle
        <answer>wrong</answer>
        <think>
        Second attempt.
        </think>
        <answer>final</answer>
        ignored suffix
        """

        normalized = normalize_trace(raw_trace)

        self.assertEqual(normalized.think_text, "First attempt.\n\nSecond attempt.")
        self.assertEqual(normalized.answer_text, "final")

    def test_curate_correct_trace_preserves_answer(self) -> None:
        raw_trace = """
        <think>
        Valid reasoning.
        </think>
        <answer>42</answer>
        """

        curated = curate_trace(
            raw_trace=raw_trace,
            is_correct=True,
            base_instructions=BASE_INSTRUCTIONS,
            clean_instructions=CLEAN_INSTRUCTIONS,
        )

        self.assertFalse(curated.uses_clean)
        self.assertIn(BASE_INSTRUCTIONS, curated.instructions)
        self.assertIn(CLEAN_INSTRUCTIONS, curated.instructions)
        self.assertIn("<answer>", curated.response)
        self.assertNotIn("<clean>", curated.response)

    def test_curate_incorrect_trace_replaces_answer_with_clean(self) -> None:
        raw_trace = """
        <think>
        This path is failing.
        </think>
        <answer>17</answer>
        """

        curated = curate_trace(
            raw_trace=raw_trace,
            is_correct=False,
            base_instructions=BASE_INSTRUCTIONS,
            clean_instructions=CLEAN_INSTRUCTIONS,
        )

        self.assertTrue(curated.uses_clean)
        self.assertIn("search is becoming confusing", curated.response)
        self.assertTrue(curated.response.strip().endswith("<clean>"))
        self.assertNotIn("<answer>", curated.response)

    def test_incorrect_trace_without_answer_requires_explicit_opt_in(self) -> None:
        raw_trace = "<think>This path never produced an answer.</think>"

        with self.assertRaises(ValueError):
            curate_trace(
                raw_trace=raw_trace,
                is_correct=False,
                base_instructions=BASE_INSTRUCTIONS,
                clean_instructions=CLEAN_INSTRUCTIONS,
            )

    def test_incorrect_trace_without_answer_can_be_explicit_clean_only_negative(self) -> None:
        raw_trace = "<think>This path is intentionally unproductive.</think>"

        curated = curate_trace(
            raw_trace=raw_trace,
            is_correct=False,
            base_instructions=BASE_INSTRUCTIONS,
            clean_instructions=CLEAN_INSTRUCTIONS,
            allow_missing_answer_for_incorrect=True,
        )

        self.assertTrue(curated.uses_clean)
        self.assertTrue(curated.response.strip().endswith("<clean>"))
        self.assertNotIn("<answer>", curated.response)

    def test_correct_trace_requires_a_final_answer(self) -> None:
        raw_trace = "<think>Reasoning only.</think>"

        with self.assertRaises(ValueError):
            curate_trace(
                raw_trace=raw_trace,
                is_correct=True,
                base_instructions=BASE_INSTRUCTIONS,
                clean_instructions=CLEAN_INSTRUCTIONS,
            )


if __name__ == "__main__":
    unittest.main()

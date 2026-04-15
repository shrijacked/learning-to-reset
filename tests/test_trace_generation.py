import unittest

from learning_to_reset.data import CountdownSample
from learning_to_reset.trace_generation import build_trace_record_from_response


class TraceGenerationTests(unittest.TestCase):
    def test_build_trace_record_from_response_marks_correct_solution(self) -> None:
        sample = CountdownSample(
            source_id="countdown-1",
            numbers=(60, 27, 19),
            target=68,
            question="Reach 68 using 60, 27, 19.",
        )

        record = build_trace_record_from_response(
            sample,
            "<think>Try the obvious sum.</think><answer>(60 + 27) - 19</answer>",
            source_model="expert-model",
        )

        self.assertEqual(record["problem"], "Reach 68 using 60, 27, 19.")
        self.assertTrue(record["is_correct"])
        self.assertEqual(record["source_id"], "countdown-1")
        self.assertEqual(record["metadata"]["source_model"], "expert-model")

    def test_build_trace_record_from_response_marks_incorrect_solution(self) -> None:
        sample = CountdownSample(
            source_id="countdown-2",
            numbers=(61, 63, 57),
            target=55,
            question="Reach 55 using 61, 63, 57.",
        )

        record = build_trace_record_from_response(
            sample,
            "<think>This is not going well.</think><answer>61 - 57</answer>",
            source_model="expert-model",
        )

        self.assertFalse(record["is_correct"])
        self.assertIn("verification_reason", record["metadata"])


if __name__ == "__main__":
    unittest.main()

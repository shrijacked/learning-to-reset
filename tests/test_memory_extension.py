import unittest

from learning_to_reset.memory_extension import (
    build_memory_clean_trajectory,
    MemoryEntry,
    RecallMemoryStore,
    build_recall_prompt,
    extract_memory_writes,
    manage_memory_clean_cycles,
)
from learning_to_reset.data import CountdownSample


class MemoryExtensionTests(unittest.TestCase):
    def test_extract_memory_writes_reads_tagged_memory_blocks(self) -> None:
        entries = extract_memory_writes(
            (
                '<memory tags="countdown,partial">60 + 27 = 87.</memory>'
                "<memory>Use 19 as the final subtraction.</memory>"
            ),
            source_id="trace-1",
        )

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].content, "60 + 27 = 87.")
        self.assertEqual(entries[0].tags, ("countdown", "partial"))
        self.assertEqual(entries[0].source_id, "trace-1")
        self.assertEqual(entries[1].tags, ())

    def test_recall_memory_store_returns_highest_overlap_entries(self) -> None:
        store = RecallMemoryStore()
        store.write("9 + 11 = 20.", tags=("other",), source_id="other")
        store.write("60 + 27 = 87.", tags=("countdown", "68"), source_id="target")

        recalled = store.recall("Reach 68 using 60, 27, and 19.", limit=1)

        self.assertEqual(len(recalled), 1)
        self.assertEqual(recalled[0].source_id, "target")
        self.assertEqual(recalled[0].content, "60 + 27 = 87.")

    def test_recall_memory_store_can_extend_from_response(self) -> None:
        store = RecallMemoryStore()

        entries = store.extend_from_response(
            '<memory tags="target">87 - 19 = 68.</memory>',
            source_id="trace-2",
        )

        self.assertEqual(entries, (MemoryEntry("87 - 19 = 68.", ("target",), "trace-2"),))
        self.assertEqual(store.entries, entries)

    def test_build_recall_prompt_includes_recalled_entries(self) -> None:
        prompt = build_recall_prompt(
            question="Reach 68 using 60, 27, 19.",
            base_instructions="Solve carefully.",
            recalled_entries=(
                MemoryEntry("60 + 27 = 87.", ("countdown",), "trace-1"),
            ),
        )

        self.assertIn("Relevant recalled memory:", prompt)
        self.assertIn("- 60 + 27 = 87.", prompt)
        self.assertIn("Reach 68 using 60, 27, 19.", prompt)

    def test_memory_clean_cycles_recall_writes_after_clean(self) -> None:
        managed = manage_memory_clean_cycles(
            question="Reach 68 using 60, 27, 19.",
            base_instructions="Solve carefully.",
            clean_instructions="Emit <clean> if needed.",
            responses=(
                (
                    '<think>Partial sum looks useful.</think>'
                    '<memory tags="countdown,68">60 + 27 = 87.</memory>'
                    "<clean>"
                ),
                "<think>Recall 87 and subtract 19.</think><answer>(60 + 27) - 19</answer>",
            ),
            max_cleans=1,
            recall_limit=1,
        )

        self.assertEqual(managed.clean_count, 1)
        self.assertEqual(managed.memory_entries, (MemoryEntry("60 + 27 = 87.", ("countdown", "68"), "segment-0"),))
        self.assertIn("Relevant recalled memory:", managed.segments[1].prompt)
        self.assertIn("- 60 + 27 = 87.", managed.segments[1].prompt)
        self.assertEqual(managed.segments[1].recalled_entries, managed.memory_entries)
        self.assertEqual(managed.final_answer, "(60 + 27) - 19")

    def test_memory_clean_trajectory_applies_clean_penalty_to_final_reward(self) -> None:
        sample = CountdownSample(
            source_id="countdown-1",
            numbers=(60, 27, 19),
            target=68,
            question="Reach 68 using 60, 27, 19.",
        )

        trajectory = build_memory_clean_trajectory(
            question=sample.question,
            sample=sample,
            responses=(
                '<memory tags="countdown">60 + 27 = 87.</memory><clean>',
                "<think>87 - 19 = 68.</think><answer>(60 + 27) - 19</answer>",
            ),
            max_cleans=1,
            clean_step_penalty=0.05,
            base_instructions="Solve carefully.",
            clean_instructions="Emit <clean> if needed.",
        )

        self.assertEqual(trajectory.managed.memory_entries[0].content, "60 + 27 = 87.")
        self.assertAlmostEqual(trajectory.segment_rewards[-1].total_reward, 1.3)
        self.assertAlmostEqual(trajectory.adjusted_total_reward, 1.25)


if __name__ == "__main__":
    unittest.main()

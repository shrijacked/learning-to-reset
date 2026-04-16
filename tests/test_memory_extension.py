import unittest

from learning_to_reset.memory_extension import (
    MemoryEntry,
    RecallMemoryStore,
    build_recall_prompt,
    extract_memory_writes,
)


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


if __name__ == "__main__":
    unittest.main()

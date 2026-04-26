import unittest

from learning_to_reset.paper_sources import (
    SCALE_PRESETS as PAPER_SOURCES_SCALE_PRESETS,
    apply_scale_preset as apply_paper_sources_scale,
    build_parser as build_paper_sources_parser,
)
from learning_to_reset.prepare_paper_artifacts import (
    SCALE_PRESETS as PREPARE_SCALE_PRESETS,
    apply_scale_preset as apply_prepare_scale,
    build_parser as build_prepare_parser,
)


class PaperScalePresetTests(unittest.TestCase):
    def test_paper_sources_scale_presets_have_pilot_and_paper_keys(self) -> None:
        self.assertIn("pilot", PAPER_SOURCES_SCALE_PRESETS)
        self.assertIn("paper", PAPER_SOURCES_SCALE_PRESETS)

    def test_paper_sources_pilot_preset_caps_dataset_sizes(self) -> None:
        pilot = PAPER_SOURCES_SCALE_PRESETS["pilot"]
        self.assertIsNotNone(pilot["max_train_rows"])
        self.assertIsNotNone(pilot["max_eval_rows"])
        self.assertIsNotNone(pilot["max_train_samples"])
        self.assertIsNotNone(pilot["max_reference_trace_rows"])

    def test_paper_sources_paper_preset_lifts_all_caps(self) -> None:
        paper = PAPER_SOURCES_SCALE_PRESETS["paper"]
        for key in (
            "max_train_rows",
            "max_eval_rows",
            "max_train_samples",
            "max_eval_samples",
            "max_reference_trace_rows",
        ):
            self.assertIsNone(paper[key], f"paper preset must lift {key}")

    def test_paper_sources_apply_scale_preset_only_fills_missing(self) -> None:
        configured = apply_paper_sources_scale(
            scale="pilot",
            overrides={
                "max_train_rows": 999,
                "max_eval_rows": None,
                "max_train_samples": None,
                "max_eval_samples": None,
                "max_reference_trace_rows": None,
            },
        )
        self.assertEqual(configured["max_train_rows"], 999)
        self.assertIsNotNone(configured["max_eval_rows"])
        self.assertIsNotNone(configured["max_reference_trace_rows"])

    def test_paper_sources_parser_accepts_scale_flag(self) -> None:
        parser = build_paper_sources_parser()
        namespace = parser.parse_args(
            ["--output-dir", "/tmp/out", "--scale", "paper"]
        )
        self.assertEqual(namespace.scale, "paper")

    def test_prepare_paper_artifacts_scale_presets_match_paper_recipe(self) -> None:
        self.assertIn("pilot", PREPARE_SCALE_PRESETS)
        self.assertIn("paper", PREPARE_SCALE_PRESETS)
        paper = PREPARE_SCALE_PRESETS["paper"]
        self.assertGreaterEqual(paper["recovery_repeat"], 4)
        self.assertEqual(paper["require_recovery_target_correct"], True)

    def test_prepare_paper_artifacts_apply_scale_preset_only_fills_missing(self) -> None:
        configured = apply_prepare_scale(
            scale="paper",
            overrides={
                "sft_val_ratio": 0.2,
                "countdown_val_ratio": None,
                "recovery_repeat": None,
                "include_recovery_examples": None,
                "require_recovery_target_correct": None,
            },
        )
        self.assertAlmostEqual(configured["sft_val_ratio"], 0.2)
        self.assertGreaterEqual(configured["recovery_repeat"], 4)
        self.assertTrue(configured["include_recovery_examples"])

    def test_prepare_paper_artifacts_parser_accepts_scale_flag(self) -> None:
        parser = build_prepare_parser()
        namespace = parser.parse_args(
            [
                "--traces", "x.jsonl",
                "--countdown-train", "y.jsonl",
                "--countdown-eval", "z.jsonl",
                "--output-dir", "/tmp/out",
                "--scale", "paper",
            ]
        )
        self.assertEqual(namespace.scale, "paper")


if __name__ == "__main__":
    unittest.main()

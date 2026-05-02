import tempfile
import unittest
from pathlib import Path

from scripts.build_figures import build_figures


class BuildFiguresTests(unittest.TestCase):
    def test_build_figures_writes_svg_pack_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "figures"

            summary = build_figures(output_dir)
            svg_files = sorted(path.name for path in output_dir.glob("*.svg"))

        self.assertIn("validity-raw-vs-reset.svg", svg_files)
        self.assertIn("hard-correctness-gap.svg", svg_files)
        self.assertIn("pipeline.svg", svg_files)
        self.assertIn("reset_0_5b", summary["metrics"])


if __name__ == "__main__":
    unittest.main()

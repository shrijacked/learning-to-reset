import tempfile
import unittest
from pathlib import Path

from learning_to_reset.model_resolver import resolve_model_name_or_path


class ModelResolverTests(unittest.TestCase):
    def test_returns_existing_local_path_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local = Path(tmp_dir) / "model-dir"
            local.mkdir()

            resolved = resolve_model_name_or_path(str(local))

        self.assertEqual(Path(resolved), local)

    def test_prefers_cached_snapshot_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir)
            older = cache_root / "models--Qwen--Qwen2.5-0.5B" / "snapshots" / "old"
            newer = cache_root / "models--Qwen--Qwen2.5-0.5B" / "snapshots" / "new"
            older.mkdir(parents=True)
            newer.mkdir(parents=True)
            older.touch()
            newer.touch()

            resolved = resolve_model_name_or_path(
                "Qwen/Qwen2.5-0.5B",
                cache_root=cache_root,
                prefer_local_cache=True,
            )

        self.assertEqual(Path(resolved), newer)

    def test_falls_back_to_original_id_when_cache_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            resolved = resolve_model_name_or_path(
                "Qwen/Qwen2.5-0.5B",
                cache_root=Path(tmp_dir),
                prefer_local_cache=True,
            )

        self.assertEqual(resolved, "Qwen/Qwen2.5-0.5B")


if __name__ == "__main__":
    unittest.main()

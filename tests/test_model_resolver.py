import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from learning_to_reset.model_resolver import resolve_model_name_or_path


class ModelResolverTests(unittest.TestCase):
    def test_returns_existing_local_path_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local = Path(tmp_dir) / "model-dir"
            local.mkdir()
            (local / "config.json").write_text("{}", encoding="utf-8")

            resolved = resolve_model_name_or_path(str(local))

        self.assertEqual(Path(resolved), local)

    def test_prefers_best_checkpoint_inside_training_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "run-output"
            best_dir = output_root / "best-checkpoint"
            best_dir.mkdir(parents=True)
            (best_dir / "config.json").write_text("{}", encoding="utf-8")

            resolved = resolve_model_name_or_path(str(output_root))

        self.assertEqual(Path(resolved), best_dir)

    def test_falls_back_to_final_checkpoint_inside_training_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "run-output"
            final_dir = output_root / "final-checkpoint"
            final_dir.mkdir(parents=True)
            (final_dir / "config.json").write_text("{}", encoding="utf-8")

            resolved = resolve_model_name_or_path(str(output_root))

        self.assertEqual(Path(resolved), final_dir)

    def test_prefers_cached_snapshot_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir)
            older = cache_root / "models--Qwen--Qwen2.5-0.5B" / "snapshots" / "old"
            newer = cache_root / "models--Qwen--Qwen2.5-0.5B" / "snapshots" / "new"
            older.mkdir(parents=True)
            newer.mkdir(parents=True)
            older_time = 1_700_000_000
            newer_time = older_time + 10
            os.utime(older, (older_time, older_time))
            os.utime(newer, (newer_time, newer_time))

            resolved = resolve_model_name_or_path(
                "Qwen/Qwen2.5-0.5B",
                cache_root=cache_root,
                prefer_local_cache=True,
            )

        self.assertEqual(Path(resolved), newer)

    def test_cached_snapshot_tie_break_ignores_ctime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir)
            older = cache_root / "models--Qwen--Qwen2.5-0.5B" / "snapshots" / "old"
            newer = cache_root / "models--Qwen--Qwen2.5-0.5B" / "snapshots" / "new"
            older.mkdir(parents=True)
            newer.mkdir(parents=True)
            tied_time = 1_700_000_000
            os.utime(older, (tied_time, tied_time))
            os.utime(newer, (tied_time, tied_time))
            original_stat = Path.stat

            def fake_stat(path: Path, *args: object, **kwargs: object) -> object:
                actual = original_stat(path, *args, **kwargs)
                if path == older:
                    return SimpleNamespace(
                        st_mode=actual.st_mode,
                        st_mtime_ns=actual.st_mtime_ns,
                        st_ctime_ns=actual.st_ctime_ns + 10_000,
                    )
                if path == newer:
                    return SimpleNamespace(
                        st_mode=actual.st_mode,
                        st_mtime_ns=actual.st_mtime_ns,
                        st_ctime_ns=actual.st_ctime_ns,
                    )
                return actual

            with patch.object(Path, "stat", fake_stat):
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

"""Resolve model identifiers to local cache snapshots when available."""

from __future__ import annotations

from pathlib import Path


def default_hf_cache_root() -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub"


def _cached_snapshot_root(model_name_or_path: str, cache_root: Path) -> Path:
    namespace, name = model_name_or_path.split("/", 1)
    return cache_root / f"models--{namespace}--{name}" / "snapshots"


def resolve_model_name_or_path(
    model_name_or_path: str,
    *,
    cache_root: Path | None = None,
    prefer_local_cache: bool = True,
) -> str:
    """Prefer an already-cached Hugging Face snapshot before hitting the network."""

    local_path = Path(model_name_or_path)
    if local_path.exists():
        return str(local_path)

    if not prefer_local_cache or "/" not in model_name_or_path:
        return model_name_or_path

    root = cache_root or default_hf_cache_root()
    snapshot_root = _cached_snapshot_root(model_name_or_path, root)
    if not snapshot_root.exists():
        return model_name_or_path

    snapshots = [path for path in snapshot_root.iterdir() if path.is_dir()]
    if not snapshots:
        return model_name_or_path

    snapshots.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(snapshots[0])

"""Data loading and preprocessing module."""

from typing import Any, List

__all__ = ["DatasetLoader", "DatasetSplits", "SplitData"]


def __getattr__(name: str) -> Any:
    """Lazy import components to avoid runpy RuntimeWarnings when executed as scripts."""
    if name in ("DatasetLoader", "DatasetSplits", "SplitData"):
        from .dataset import DatasetLoader, DatasetSplits, SplitData
        if name == "DatasetLoader":
            return DatasetLoader
        elif name == "DatasetSplits":
            return DatasetSplits
        return SplitData
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return __all__

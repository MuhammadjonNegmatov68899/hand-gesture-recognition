"""Gesture classification models module."""

from typing import Any, List

__all__ = ["RandomForestGestureClassifier", "MLPGestureClassifier"]


def __getattr__(name: str) -> Any:
    """Lazy import models to avoid runpy RuntimeWarnings when executed as scripts."""
    if name == "RandomForestGestureClassifier":
        from .random_forest import RandomForestGestureClassifier
        return RandomForestGestureClassifier
    elif name == "MLPGestureClassifier":
        from .mlp import MLPGestureClassifier
        return MLPGestureClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return __all__

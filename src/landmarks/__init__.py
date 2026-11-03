"""Landmark detection and normalization module."""

from typing import Any, List

__all__ = ["HandDetector", "HandLandmarkData", "LandmarkNormalizer"]


def __getattr__(name: str) -> Any:
    """Lazy import components to avoid runpy RuntimeWarnings when executed as scripts."""
    if name in ("HandDetector", "HandLandmarkData"):
        from .hand_detector import HandDetector, HandLandmarkData
        if name == "HandDetector":
            return HandDetector
        return HandLandmarkData
    elif name == "LandmarkNormalizer":
        from .normalizer import LandmarkNormalizer
        return LandmarkNormalizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return __all__

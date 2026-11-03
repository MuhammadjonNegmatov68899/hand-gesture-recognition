"""Model evaluation and benchmarking module."""

from typing import Any, List

__all__ = [
    "EndToEndLatencyBreakdown",
    "EvaluationReport",
    "LatencyStats",
    "benchmark_classifier_latency",
    "benchmark_end_to_end_latency",
    "compute_latency_stats",
    "evaluate_model",
]


def __getattr__(name: str) -> Any:
    """Lazy import components to avoid runpy RuntimeWarnings when executed as scripts."""
    from . import evaluate
    if hasattr(evaluate, name):
        return getattr(evaluate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return __all__

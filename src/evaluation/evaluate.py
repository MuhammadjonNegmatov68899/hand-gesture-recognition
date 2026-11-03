"""Model evaluation and latency benchmarking for hand gesture recognition.

This module provides standard evaluation metrics and CPU latency benchmarking:
1. Classification metrics:
   - Accuracy
   - Macro F1-score
   - Macro Precision and Recall
   - Confusion Matrix
   - Model size (in KB)
   - Training time (in seconds)
2. Latency Benchmarking:
   - Classifier-only inference latency (Mean, P95, FPS)
   - End-to-end pipeline latency (Mean, P95, FPS):
     T_total = T_MediaPipe + T_normalization + T_classifier
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import sys
import time

# Ensure project root is on sys.path for direct script execution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import threadpoolctl

from src.data.dataset import DatasetLoader, DatasetSplits
from src.landmarks.hand_detector import HandDetector
from src.landmarks.normalizer import LandmarkNormalizer
from src.models.mlp import MLPGestureClassifier
from src.models.random_forest import RandomForestGestureClassifier


@dataclass
class LatencyStats:
    """Statistical summary of execution latency."""

    mean_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    fps: float
    iterations: int


@dataclass
class EndToEndLatencyBreakdown:
    """Latency breakdown for end-to-end pipeline: T_total = T_mp + T_norm + T_clf."""

    mediapipe: LatencyStats
    normalization: LatencyStats
    classifier: LatencyStats
    total: LatencyStats


@dataclass
class EvaluationReport:
    """Complete evaluation report container."""

    model_name: str
    model_size_kb: float
    training_time_sec: Optional[float]
    accuracy: float
    f1_macro: float
    precision_macro: float
    recall_macro: float
    classes: List[str]
    confusion_matrix: List[List[int]]
    classifier_latency: LatencyStats
    end_to_end_latency: Optional[EndToEndLatencyBreakdown] = None
    cpu_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return asdict(self)

    def print_summary(self) -> None:
        """Print human-readable formatted summary table."""
        print("\n" + "=" * 65)
        print(f"EVALUATION REPORT: {self.model_name}")
        print("=" * 65)
        print(f"Model Size:        {self.model_size_kb:.2f} KB")
        if self.training_time_sec is not None:
            print(f"Training Time:     {self.training_time_sec:.4f} s")
        print("-" * 65)
        print("Classification Metrics (Test Set):")
        print(f" - Accuracy:        {self.accuracy * 100:.2f}%")
        print(f" - Macro F1-Score:  {self.f1_macro:.4f}")
        print(f" - Macro Precision: {self.precision_macro:.4f}")
        print(f" - Macro Recall:    {self.recall_macro:.4f}")
        print(f" - Classes ({len(self.classes)}): {', '.join(self.classes)}")
        print("-" * 65)
        print("Classifier-Only Inference Latency:")
        print(f" - Mean Latency:    {self.classifier_latency.mean_ms:.3f} ms")
        print(f" - P95 Latency:     {self.classifier_latency.p95_ms:.3f} ms")
        print(f" - P99 Latency:     {self.classifier_latency.p99_ms:.3f} ms")
        print(f" - Throughput:      {self.classifier_latency.fps:.1f} FPS")

        if self.end_to_end_latency is not None:
            print("-" * 65)
            print("End-to-End Pipeline Latency (T_total = T_mp + T_norm + T_clf):")
            e2e = self.end_to_end_latency
            print(f" - MediaPipe (T_mp):      {e2e.mediapipe.mean_ms:.3f} ms  (P95: {e2e.mediapipe.p95_ms:.3f} ms)")
            print(f" - Normalization (T_norm): {e2e.normalization.mean_ms:.3f} ms  (P95: {e2e.normalization.p95_ms:.3f} ms)")
            print(f" - Classifier (T_clf):    {e2e.classifier.mean_ms:.3f} ms  (P95: {e2e.classifier.p95_ms:.3f} ms)")
            print(f" -> Total (T_total):      {e2e.total.mean_ms:.3f} ms  (P95: {e2e.total.p95_ms:.3f} ms | {e2e.total.fps:.1f} FPS)")
        print("=" * 65 + "\n")


def compute_latency_stats(latencies_ms: Sequence[float]) -> LatencyStats:
    """Calculate mean, p95, p99, min, max, and FPS from a list of latencies in ms."""
    arr = np.asarray(latencies_ms, dtype=np.float64)
    if len(arr) == 0:
        raise ValueError("Cannot calculate latency stats on empty list.")
    mean_val = float(np.mean(arr))
    p95_val = float(np.percentile(arr, 95))
    p99_val = float(np.percentile(arr, 99))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    fps = 1000.0 / mean_val if mean_val > 0 else 0.0

    return LatencyStats(
        mean_ms=mean_val,
        p95_ms=p95_val,
        p99_ms=p99_val,
        min_ms=min_val,
        max_ms=max_val,
        fps=fps,
        iterations=len(arr),
    )


def benchmark_classifier_latency(
    model: Union[RandomForestGestureClassifier, MLPGestureClassifier],
    sample_features: Optional[np.ndarray] = None,
    n_iterations: int = 500,
    warmup_iterations: int = 50,
) -> LatencyStats:
    """Benchmark inference latency for classifier alone on single feature vector.

    Args:
        model: Trained classifier instance.
        sample_features: Array of shape (63,) or (1, 63). If None, zeros are used.
        n_iterations: Number of timed benchmark iterations.
        warmup_iterations: Warmup runs to prime caches and runtime.

    Returns:
        LatencyStats summary.
    """
    if sample_features is None:
        sample_features = np.zeros((1, 63), dtype=np.float32)
    elif sample_features.ndim == 1:
        sample_features = sample_features.reshape(1, 63)

    # Warmup
    for _ in range(warmup_iterations):
        _ = model.predict(sample_features)

    # Timed runs
    latencies: List[float] = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = model.predict(sample_features)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    return compute_latency_stats(latencies)


def benchmark_end_to_end_latency(
    detector: HandDetector,
    normalizer: LandmarkNormalizer,
    model: Union[RandomForestGestureClassifier, MLPGestureClassifier],
    benchmark_image: np.ndarray,
    n_iterations: int = 100,
    warmup_iterations: int = 10,
) -> EndToEndLatencyBreakdown:
    """Benchmark end-to-end landmark recognition pipeline on a real hand image.

    Formula: T_total = T_MediaPipe + T_normalization + T_classifier.
    Excludes image loading and webcam startup IO.
    """
    if benchmark_image is None or benchmark_image.size == 0:
        raise ValueError("A valid non-empty benchmark image must be provided.")

    # 1. Verification run: Verify that MediaPipe detects at least one hand
    initial_res = detector.detect(benchmark_image)
    initial_hands = detector.extract_landmarks(initial_res)
    if not initial_hands:
        raise RuntimeError(
            "MediaPipe Hand Landmarker detected 0 hands in the provided benchmark image.\n"
            "To measure realistic landmark inference, normalization, and classification latency, "
            "the benchmark image must contain at least one clearly visible hand.\n"
            "Blank, dark, or hand-less frames are rejected because MediaPipe exits early without "
            "running the landmark regression submodel."
        )

    # Warmup runs
    for _ in range(warmup_iterations):
        res = detector.detect(benchmark_image)
        hands = detector.extract_landmarks(res)
        if not hands:
            raise RuntimeError("Hand tracking lost during benchmark warmup.")
        feat = normalizer.normalize(hands[0].flat_landmarks)
        _ = model.predict(feat)

    t_mp_list: List[float] = []
    t_norm_list: List[float] = []
    t_clf_list: List[float] = []
    t_total_list: List[float] = []

    for _ in range(n_iterations):
        # 1. T_MediaPipe (palm detection + landmark tracking)
        t0 = time.perf_counter()
        res = detector.detect(benchmark_image)
        hands = detector.extract_landmarks(res)
        t1 = time.perf_counter()
        t_mp = (t1 - t0) * 1000.0

        if not hands:
            raise RuntimeError("Hand tracking lost during timed benchmark iteration.")

        # 2. T_normalization (wrist translation + anatomical scale reference)
        t2 = time.perf_counter()
        feat = normalizer.normalize(hands[0].flat_landmarks)
        t3 = time.perf_counter()
        t_norm = (t3 - t2) * 1000.0

        # 3. T_classifier (MLP or Random Forest prediction)
        t4 = time.perf_counter()
        _ = model.predict(feat)
        t5 = time.perf_counter()
        t_clf = (t5 - t4) * 1000.0

        t_total = t_mp + t_norm + t_clf

        t_mp_list.append(t_mp)
        t_norm_list.append(t_norm)
        t_clf_list.append(t_clf)
        t_total_list.append(t_total)

    return EndToEndLatencyBreakdown(
        mediapipe=compute_latency_stats(t_mp_list),
        normalization=compute_latency_stats(t_norm_list),
        classifier=compute_latency_stats(t_clf_list),
        total=compute_latency_stats(t_total_list),
    )


def load_and_validate_model(
    model_path: Union[str, Path],
    expected_type: Optional[str] = None,
) -> Union[RandomForestGestureClassifier, MLPGestureClassifier]:
    """Load serialized model and validate against expected architecture."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found at: {path.resolve()}")

    data = joblib.load(path)
    model_type_str = data.get("model_type") if isinstance(data, dict) else None

    if expected_type == "rf":
        if model_type_str and model_type_str != "RandomForestGestureClassifier":
            raise ValueError(
                f"Model type mismatch: CLI requested --model-type 'rf', but '{path.name}' "
                f"contains a '{model_type_str}' model."
            )
        return RandomForestGestureClassifier.load(path)
    elif expected_type == "mlp":
        if model_type_str and model_type_str != "MLPGestureClassifier":
            raise ValueError(
                f"Model type mismatch: CLI requested --model-type 'mlp', but '{path.name}' "
                f"contains a '{model_type_str}' model."
            )
        return MLPGestureClassifier.load(path)
    else:
        # Autodetect
        if model_type_str == "RandomForestGestureClassifier":
            return RandomForestGestureClassifier.load(path)
        elif model_type_str == "MLPGestureClassifier":
            return MLPGestureClassifier.load(path)
        raise ValueError(
            f"Could not determine model type from '{path.name}'. "
            f"Please specify --model-type 'rf' or 'mlp'."
        )


def evaluate_model(
    model: Union[RandomForestGestureClassifier, MLPGestureClassifier],
    X_test: np.ndarray,
    y_test: Sequence[Any],
    detector: Optional[HandDetector] = None,
    benchmark_image: Optional[np.ndarray] = None,
    normalizer: Optional[LandmarkNormalizer] = None,
    n_latency_iterations: int = 500,
    benchmark_e2e: bool = False,
    e2e_iterations: Optional[int] = None,
    num_threads: int = 1,
) -> EvaluationReport:
    """Full evaluation of a gesture classification model on official test data.

    Calculates:
    - Accuracy
    - Macro F1
    - Precision and recall
    - Confusion matrix
    - Model size in KB
    - Training time
    - Classifier inference latency (mean, P95, FPS)
    - Optional end-to-end pipeline latency breakdown
    """
    if not model.is_fitted:
        raise RuntimeError("Model must be trained before evaluation.")

    normalizer = normalizer or LandmarkNormalizer()

    # Enforce CPU thread settings:
    # 1. For BLAS (MLP), use threadpoolctl
    # 2. For joblib (Random Forest), explicitly set clf.n_jobs
    original_n_jobs = getattr(getattr(model, "clf", None), "n_jobs", None)
    if hasattr(model, "clf") and hasattr(model.clf, "n_jobs"):
        model.clf.n_jobs = num_threads

    try:
        with threadpoolctl.threadpool_limits(limits=num_threads):
            preds = model.predict(X_test)
            acc = float(accuracy_score(y_test, preds))
            f1_mac = float(f1_score(y_test, preds, average="macro", zero_division=0))
            prec_mac = float(precision_score(y_test, preds, average="macro", zero_division=0))
            rec_mac = float(recall_score(y_test, preds, average="macro", zero_division=0))
            cm = confusion_matrix(y_test, preds, labels=model.classes_)

            # Benchmark classifier latency on test sample
            sample_feat = X_test[0] if len(X_test) > 0 else np.zeros(63, dtype=np.float32)
            clf_latency = benchmark_classifier_latency(
                model=model,
                sample_features=sample_feat,
                n_iterations=n_latency_iterations,
            )

            # Benchmark end-to-end latency if requested and detector provided
            e2e_breakdown = None
            if benchmark_e2e and detector is not None:
                if benchmark_image is None:
                    raise ValueError(
                        "End-to-end latency benchmark requires a valid hand image. "
                        "Please provide benchmark_image containing at least one detectable hand."
                    )
                n_e2e = e2e_iterations if e2e_iterations is not None else n_latency_iterations
                e2e_breakdown = benchmark_end_to_end_latency(
                    detector=detector,
                    normalizer=normalizer,
                    model=model,
                    benchmark_image=benchmark_image,
                    n_iterations=n_e2e,
                )
    finally:
        if hasattr(model, "clf") and hasattr(model.clf, "n_jobs") and original_n_jobs is not None:
            model.clf.n_jobs = original_n_jobs

    report = EvaluationReport(
        model_name=type(model).__name__,
        model_size_kb=model.get_model_size_kb(),
        training_time_sec=model.training_time_sec,
        accuracy=acc,
        f1_macro=f1_mac,
        precision_macro=prec_mac,
        recall_macro=rec_mac,
        classes=[str(c) for c in model.classes_],
        confusion_matrix=cm.tolist(),
        classifier_latency=clf_latency,
        end_to_end_latency=e2e_breakdown,
        cpu_info={
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "threads_configured": num_threads,
        },
    )
    return report


def main() -> None:
    """CLI entry point for model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Hand Gesture Recognition Landmark Models."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model (.joblib file)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["rf", "mlp"],
        default="rf",
        help="Model architecture: 'rf' (Random Forest) or 'mlp'",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to dataset directory containing official splits (default: data)",
    )
    parser.add_argument(
        "--benchmark-e2e",
        action="store_true",
        help="Run end-to-end latency benchmark including MediaPipe HandLandmarker",
    )
    parser.add_argument(
        "--benchmark-image",
        type=str,
        default=None,
        help="Path to real image containing a hand for end-to-end benchmark",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Number of latency benchmark iterations (default: 500)",
    )
    parser.add_argument(
        "--e2e-iterations",
        type=int,
        default=None,
        help="Number of iterations for end-to-end benchmark (defaults to --iterations if omitted)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of CPU threads to configure (default: 1)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save JSON evaluation report",
    )
    args = parser.parse_args()

    # Load dataset
    loader = DatasetLoader(normalize=True)
    try:
        splits = loader.load_dataset(args.data_dir)
    except FileNotFoundError as err:
        print(f"\n[EVALUATION BLOCKED] Official dataset not available:\n{err}", file=sys.stderr)
        print("\nModel evaluation is PENDING until Person 1 provides the shared dataset.\n", file=sys.stderr)
        sys.exit(2)

    # Load and validate model against expected architecture
    try:
        model = load_and_validate_model(args.model_path, expected_type=args.model_type)
    except Exception as err:
        print(f"[Error] Failed to load model: {err}", file=sys.stderr)
        sys.exit(1)

    benchmark_img = None
    detector = None
    if args.benchmark_e2e:
        if not args.benchmark_image:
            print(
                "[Error] --benchmark-e2e requires a real hand image supplied via "
                "--benchmark-image <path/to/hand_image.jpg>.\n"
                "Blank, dark, or synthetic frames are rejected because realistic latency "
                "measurement requires executing the full palm detection and landmark regression pipeline.",
                file=sys.stderr,
            )
            sys.exit(1)

        img_path = Path(args.benchmark_image)
        if not img_path.is_file():
            print(f"[Error] Benchmark image not found at: {img_path.resolve()}", file=sys.stderr)
            sys.exit(1)

        benchmark_img = cv2.imread(str(img_path))
        if benchmark_img is None:
            print(f"[Error] Failed to decode image file at: {img_path.resolve()}", file=sys.stderr)
            sys.exit(1)

        detector = HandDetector()

    try:
        report = evaluate_model(
            model=model,
            X_test=splits.test.X,
            y_test=splits.test.y,
            detector=detector,
            benchmark_image=benchmark_img,
            n_latency_iterations=args.iterations,
            benchmark_e2e=args.benchmark_e2e,
            e2e_iterations=args.e2e_iterations,
            num_threads=args.threads,
        )

        report.print_summary()

        if args.output_json:
            out_p = Path(args.output_json)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"Saved evaluation report to {out_p.resolve()}")
    finally:
        if detector:
            detector.close()


if __name__ == "__main__":
    main()

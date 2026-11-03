"""Random Forest classifier for hand gesture landmark recognition."""

from __future__ import annotations

import io
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure project root is on sys.path for direct script execution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


class RandomForestGestureClassifier:
    """Random Forest classifier for 63-dimensional normalized hand gesture landmarks."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        n_jobs: Optional[int] = None,
        class_weight: Optional[Union[str, Dict[Any, float]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Random Forest classifier.

        Args:
            n_estimators: Number of trees in the forest (default: 100).
            max_depth: Maximum depth of the tree (default: None for full expansion).
            min_samples_split: Minimum number of samples required to split an internal node.
            min_samples_leaf: Minimum number of samples required to be at a leaf node.
            random_state: Random state for deterministic reproducibility (default: 42).
            n_jobs: Number of CPU threads to use (-1 means all available cores).
            class_weight: Optional class weights balancing scheme.
            **kwargs: Additional parameters forwarded to sklearn's RandomForestClassifier.
        """
        self.params: Dict[str, Any] = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "class_weight": class_weight,
            **kwargs,
        }
        self.clf = RandomForestClassifier(**self.params)
        self.training_time_sec: Optional[float] = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been trained."""
        return self._is_fitted

    @property
    def classes_(self) -> np.ndarray:
        """Classes known to the classifier."""
        if not self._is_fitted:
            raise RuntimeError("Classifier has not been fitted yet.")
        return self.clf.classes_

    def _validate_features(self, X: np.ndarray) -> np.ndarray:
        """Validate input features array."""
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] != 63:
                raise ValueError(
                    f"Expected 63 landmark features, got shape {arr.shape}."
                )
            arr = arr.reshape(1, 63)
        elif arr.ndim == 2:
            if arr.shape[1] != 63:
                raise ValueError(
                    f"Expected 63 landmark features per sample, got shape {arr.shape}."
                )
        else:
            raise ValueError(
                f"Features array must be 1D (63,) or 2D (N, 63), got ndim={arr.ndim}."
            )

        if not np.isfinite(arr).all():
            raise ValueError("Input features contain NaN, Inf, or invalid numbers.")
        return arr

    def fit(
        self,
        X_train: np.ndarray,
        y_train: Sequence[Any],
        **fit_params: Any,
    ) -> RandomForestGestureClassifier:
        """Train the Random Forest model on feature vectors.

        Args:
            X_train: Array of shape (N, 63) containing normalized landmark features.
            y_train: Target labels of length N.
            **fit_params: Optional arguments passed to underlying fit().

        Returns:
            self
        """
        X_arr = self._validate_features(X_train)
        y_arr = np.asarray(y_train)

        if len(X_arr) != len(y_arr):
            raise ValueError(
                f"Features count ({len(X_arr)}) and labels count ({len(y_arr)}) must match."
            )
        if len(X_arr) == 0:
            raise ValueError("Training data cannot be empty.")

        t0 = time.perf_counter()
        self.clf.fit(X_arr, y_arr, **fit_params)
        self.training_time_sec = time.perf_counter() - t0
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict gesture labels for input feature vectors.

        Args:
            X: Array of shape (N, 63) or (63,) normalized landmark features.

        Returns:
            1D array of predicted class labels.
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier has not been trained yet. Call fit() first.")
        X_arr = self._validate_features(X)
        preds = self.clf.predict(X_arr)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for input feature vectors.

        Args:
            X: Array of shape (N, 63) or (63,) normalized landmark features.

        Returns:
            2D array of class probabilities of shape (N, num_classes).
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier has not been trained yet. Call fit() first.")
        X_arr = self._validate_features(X)
        return self.clf.predict_proba(X_arr)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: Sequence[Any],
    ) -> Dict[str, Any]:
        """Evaluate the model on test features and labels.

        Args:
            X_test: Test features of shape (N, 63).
            y_test: Ground truth labels.

        Returns:
            Dictionary containing accuracy, macro F1, and confusion matrix.
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier has not been trained yet.")

        X_arr = self._validate_features(X_test)
        y_arr = np.asarray(y_test)

        preds = self.clf.predict(X_arr)
        acc = float(accuracy_score(y_arr, preds))
        f1_macro = float(f1_score(y_arr, preds, average="macro", zero_division=0))
        cm = confusion_matrix(y_arr, preds, labels=self.clf.classes_)

        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "classes": [str(c) for c in self.clf.classes_],
            "confusion_matrix": cm.tolist(),
        }

    def save(self, filepath: Union[str, Path]) -> Path:
        """Serialize and save the trained model to disk.

        Args:
            filepath: Destination path (e.g. models/random_forest.joblib).

        Returns:
            Resolved Path of the saved file.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model. Call fit() first.")

        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_type": "RandomForestGestureClassifier",
            "params": self.params,
            "training_time_sec": self.training_time_sec,
            "classifier": self.clf,
        }
        joblib.dump(payload, save_path)
        return save_path.resolve()

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> RandomForestGestureClassifier:
        """Load a serialized model from disk.

        Args:
            filepath: Path to the serialized joblib file.

        Returns:
            Loaded RandomForestGestureClassifier instance.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found at: {path.resolve()}")

        data = joblib.load(path)
        if not isinstance(data, dict) or "classifier" not in data:
            # Handle direct classifier instance dump
            instance = cls()
            instance.clf = data
            instance._is_fitted = True
            return instance

        instance = cls(**data.get("params", {}))
        instance.clf = data["classifier"]
        instance.training_time_sec = data.get("training_time_sec", None)
        instance._is_fitted = True
        return instance

    def get_model_size_kb(self) -> float:
        """Calculate in-memory serialized model size in Kilobytes."""
        if not self._is_fitted:
            return 0.0
        buf = io.BytesIO()
        joblib.dump(self.clf, buf)
        return len(buf.getvalue()) / 1024.0


def main() -> None:
    """CLI entry point to train and save the Random Forest model."""
    import argparse
    import sys
    from src.data.dataset import DatasetLoader

    parser = argparse.ArgumentParser(description="Train Random Forest Hand Gesture Classifier")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing official dataset")
    parser.add_argument("--output", type=str, default="models/random_forest.joblib", help="Output model path")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees (default: 100)")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum tree depth (default: None)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="CPU jobs (default: -1)")
    args = parser.parse_args()

    loader = DatasetLoader(normalize=True)
    try:
        splits = loader.load_dataset(args.data_dir)
    except FileNotFoundError as e:
        print(f"\n[DatasetPending] Model training cannot proceed:\n{e}", file=sys.stderr)
        print("\nRandom Forest training is PENDING until Person 1 provides the shared dataset.\n", file=sys.stderr)
        sys.exit(2)

    print("=" * 60)
    print("Training Random Forest Hand Gesture Classifier")
    print("=" * 60)
    print(splits.summary())

    rf = RandomForestGestureClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    rf.fit(splits.train.X, splits.train.y)
    print(f"\nTraining completed in {rf.training_time_sec:.4f} seconds.")

    # Evaluate on test set
    metrics = rf.evaluate(splits.test.X, splits.test.y)
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Test Macro F1: {metrics['f1_macro']:.4f}")

    saved_path = rf.save(args.output)
    print(f"Model saved to: {saved_path} ({rf.get_model_size_kb():.2f} KB)")


if __name__ == "__main__":
    main()


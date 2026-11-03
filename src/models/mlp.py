"""Multilayer Perceptron (MLP) classifier for hand gesture landmark recognition."""

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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier


class MLPGestureClassifier:
    """Multilayer Perceptron classifier for 63-dimensional normalized hand gesture landmarks."""

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 0.0001,
        learning_rate_init: float = 0.001,
        max_iter: int = 300,
        early_stopping: bool = False,
        n_iter_no_change: int = 15,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initialize the MLP classifier.

        Default architecture: Two hidden layers with 64 and 32 units, ReLU activation,
        and Adam optimizer for lightweight CPU inference.

        Args:
            hidden_layer_sizes: Tuple specifying neurons per hidden layer (default: (64, 32)).
            activation: Activation function ('relu', 'tanh', 'logistic').
            solver: Optimization solver ('adam', 'sgd', 'lbfgs').
            alpha: L2 regularization penalty parameter.
            learning_rate_init: Initial learning rate.
            max_iter: Maximum number of training epochs.
            early_stopping: Whether to terminate training when validation score halts improvement.
            n_iter_no_change: Maximum epochs with no improvement before stopping.
            random_state: Random state seed for reproducible weight initialization.
            **kwargs: Additional parameters forwarded to sklearn's MLPClassifier.
        """
        self.params: Dict[str, Any] = {
            "hidden_layer_sizes": tuple(hidden_layer_sizes),
            "activation": activation,
            "solver": solver,
            "alpha": alpha,
            "learning_rate_init": learning_rate_init,
            "max_iter": max_iter,
            "early_stopping": early_stopping,
            "n_iter_no_change": n_iter_no_change,
            "random_state": random_state,
            **kwargs,
        }
        self.clf = MLPClassifier(**self.params)
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
    ) -> MLPGestureClassifier:
        """Train the MLP model on feature vectors.

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
        return self.clf.predict(X_arr)

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
            filepath: Destination path (e.g. models/mlp.joblib).

        Returns:
            Resolved Path of the saved file.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model. Call fit() first.")

        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_type": "MLPGestureClassifier",
            "params": self.params,
            "training_time_sec": self.training_time_sec,
            "classifier": self.clf,
        }
        joblib.dump(payload, save_path)
        return save_path.resolve()

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> MLPGestureClassifier:
        """Load a serialized model from disk.

        Args:
            filepath: Path to the serialized joblib file.

        Returns:
            Loaded MLPGestureClassifier instance.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found at: {path.resolve()}")

        data = joblib.load(path)
        if not isinstance(data, dict) or "classifier" not in data:
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
    """CLI entry point to train and save the MLP model."""
    import argparse
    import sys
    from src.data.dataset import DatasetLoader

    parser = argparse.ArgumentParser(description="Train MLP Hand Gesture Classifier")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing official dataset")
    parser.add_argument("--output", type=str, default="models/mlp.joblib", help="Output model path")
    parser.add_argument("--hidden-layers", type=str, default="64,32", help="Comma-separated layer sizes (default: 64,32)")
    parser.add_argument("--max-iter", type=int, default=300, help="Max iterations (default: 300)")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Initial learning rate (default: 0.001)")
    parser.add_argument("--alpha", type=float, default=0.0001, help="L2 penalty alpha (default: 0.0001)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    layers = tuple(int(x.strip()) for x in args.hidden_layers.split(","))

    loader = DatasetLoader(normalize=True)
    try:
        splits = loader.load_dataset(args.data_dir)
    except FileNotFoundError as e:
        print(f"\n[DatasetPending] Model training cannot proceed:\n{e}", file=sys.stderr)
        print("\nMLP training is PENDING until Person 1 provides the shared dataset.\n", file=sys.stderr)
        sys.exit(2)

    print("=" * 60)
    print("Training MLP Hand Gesture Classifier")
    print("=" * 60)
    print(splits.summary())

    mlp = MLPGestureClassifier(
        hidden_layer_sizes=layers,
        max_iter=args.max_iter,
        learning_rate_init=args.learning_rate,
        alpha=args.alpha,
        random_state=args.random_state,
    )
    mlp.fit(splits.train.X, splits.train.y)
    print(f"\nTraining completed in {mlp.training_time_sec:.4f} seconds.")

    # Evaluate on test set
    metrics = mlp.evaluate(splits.test.X, splits.test.y)
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Test Macro F1: {metrics['f1_macro']:.4f}")

    saved_path = mlp.save(args.output)
    print(f"Model saved to: {saved_path} ({mlp.get_model_size_kb():.2f} KB)")


if __name__ == "__main__":
    main()


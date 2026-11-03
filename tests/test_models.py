"""Unit tests for Random Forest and MLP landmark classifiers."""

from pathlib import Path
import numpy as np
import pytest

from src.models.mlp import MLPGestureClassifier
from src.models.random_forest import RandomForestGestureClassifier


@pytest.fixture
def synthetic_classification_data():
    """Deterministic synthetic multiclass dataset for testing only."""
    rng = np.random.RandomState(42)
    n_samples_per_class = 20
    classes = ["fist", "palm", "peace"]
    X_list = []
    y_list = []

    for i, c in enumerate(classes):
        # Create distinct clusters in 63-dimensional space
        cluster_center = np.zeros(63, dtype=np.float32)
        cluster_center[i * 10 : (i + 1) * 10] = 2.0
        features = cluster_center + rng.normal(0, 0.2, size=(n_samples_per_class, 63)).astype(np.float32)
        X_list.append(features)
        y_list.extend([c] * n_samples_per_class)

    X = np.vstack(X_list)
    y = np.array(y_list)

    # Train / test split (45 train, 15 test)
    indices = rng.permutation(len(X))
    train_idx, test_idx = indices[:45], indices[45:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def test_random_forest_training_and_prediction(synthetic_classification_data):
    X_train, y_train, X_test, y_test = synthetic_classification_data
    rf = RandomForestGestureClassifier(n_estimators=20, random_state=42)

    assert not rf.is_fitted
    with pytest.raises(RuntimeError):
        rf.predict(X_test)

    rf.fit(X_train, y_train)
    assert rf.is_fitted
    assert rf.training_time_sec is not None and rf.training_time_sec > 0
    assert set(rf.classes_) == {"fist", "palm", "peace"}

    preds = rf.predict(X_test)
    assert len(preds) == len(y_test)
    assert all(p in rf.classes_ for p in preds)

    # Test single sample prediction
    single_pred = rf.predict(X_test[0])
    assert single_pred.shape == (1,)
    assert single_pred[0] == preds[0]

    # Predict proba
    probs = rf.predict_proba(X_test)
    assert probs.shape == (len(y_test), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    # Evaluate
    metrics = rf.evaluate(X_test, y_test)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["accuracy"] >= 0.0


def test_random_forest_serialization(synthetic_classification_data, tmp_path: Path):
    X_train, y_train, X_test, _ = synthetic_classification_data
    rf = RandomForestGestureClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)

    expected_preds = rf.predict(X_test)
    model_file = tmp_path / "rf_model.joblib"
    saved_path = rf.save(model_file)
    assert saved_path.is_file()
    assert rf.get_model_size_kb() > 0

    # Load model
    loaded_rf = RandomForestGestureClassifier.load(model_file)
    assert loaded_rf.is_fitted
    loaded_preds = loaded_rf.predict(X_test)
    np.testing.assert_array_equal(expected_preds, loaded_preds)


def test_random_forest_feature_validation():
    rf = RandomForestGestureClassifier(n_estimators=5, random_state=42)
    X_valid = np.zeros((10, 63), dtype=np.float32)
    y_valid = ["a", "b"] * 5
    rf.fit(X_valid, y_valid)

    # Wrong feature dimension
    with pytest.raises(ValueError, match="Expected 63"):
        rf.predict(np.zeros((5, 30)))

    # NaN input
    bad_features = np.zeros((5, 63))
    bad_features[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN, Inf"):
        rf.predict(bad_features)


def test_mlp_training_and_prediction(synthetic_classification_data):
    X_train, y_train, X_test, y_test = synthetic_classification_data
    mlp = MLPGestureClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=50,
        random_state=42,
    )

    assert not mlp.is_fitted
    with pytest.raises(RuntimeError):
        mlp.predict(X_test)

    mlp.fit(X_train, y_train)
    assert mlp.is_fitted
    assert mlp.training_time_sec is not None and mlp.training_time_sec > 0
    assert set(mlp.classes_) == {"fist", "palm", "peace"}

    preds = mlp.predict(X_test)
    assert len(preds) == len(y_test)
    assert all(p in mlp.classes_ for p in preds)

    # Probabilities
    probs = mlp.predict_proba(X_test)
    assert probs.shape == (len(y_test), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    # Single sample prediction
    single_pred = mlp.predict(X_test[0])
    assert single_pred.shape == (1,)
    assert single_pred[0] == preds[0]

    # Evaluate
    metrics = mlp.evaluate(X_test, y_test)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["accuracy"] >= 0.0


def test_mlp_serialization(synthetic_classification_data, tmp_path: Path):
    X_train, y_train, X_test, _ = synthetic_classification_data
    mlp = MLPGestureClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=20,
        random_state=42,
    )
    mlp.fit(X_train, y_train)

    expected_preds = mlp.predict(X_test)
    model_file = tmp_path / "mlp_model.joblib"
    saved_path = mlp.save(model_file)
    assert saved_path.is_file()
    assert mlp.get_model_size_kb() > 0

    # Load model
    loaded_mlp = MLPGestureClassifier.load(model_file)
    assert loaded_mlp.is_fitted
    loaded_preds = loaded_mlp.predict(X_test)
    np.testing.assert_array_equal(expected_preds, loaded_preds)


def test_mlp_feature_validation():
    mlp = MLPGestureClassifier(max_iter=10, random_state=42)
    X_valid = np.zeros((10, 63), dtype=np.float32)
    y_valid = ["a", "b"] * 5
    mlp.fit(X_valid, y_valid)

    # Wrong feature dimension
    with pytest.raises(ValueError, match="Expected 63"):
        mlp.predict(np.zeros((5, 50)))

    # Inf input
    bad_features = np.zeros((5, 63))
    bad_features[1, 1] = np.inf
    with pytest.raises(ValueError, match="NaN, Inf"):
        mlp.predict(bad_features)


def test_latency_stats_computation():
    from src.evaluation.evaluate import compute_latency_stats

    # Latencies in ms: [1.0, 2.0, 3.0, 4.0, 5.0]
    latencies = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = compute_latency_stats(latencies)

    assert stats.mean_ms == 3.0
    assert stats.min_ms == 1.0
    assert stats.max_ms == 5.0
    assert stats.iterations == 5
    assert stats.fps == pytest.approx(1000.0 / 3.0, rel=1e-3)
    assert stats.p95_ms >= 4.0

    with pytest.raises(ValueError, match="empty"):
        compute_latency_stats([])


def test_classifier_latency_benchmarking(synthetic_classification_data):
    from src.evaluation.evaluate import benchmark_classifier_latency

    X_train, y_train, _, _ = synthetic_classification_data
    rf = RandomForestGestureClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)

    stats = benchmark_classifier_latency(
        model=rf,
        sample_features=X_train[0],
        n_iterations=20,
        warmup_iterations=5,
    )

    assert stats.iterations == 20
    assert stats.mean_ms > 0
    assert stats.fps > 0


def test_full_evaluation_pipeline(synthetic_classification_data):
    from src.evaluation.evaluate import evaluate_model

    X_train, y_train, X_test, y_test = synthetic_classification_data
    rf = RandomForestGestureClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)

    report = evaluate_model(
        model=rf,
        X_test=X_test,
        y_test=y_test,
        n_latency_iterations=25,
        benchmark_e2e=False,
        num_threads=1,
    )

    assert report.model_name == "RandomForestGestureClassifier"
    assert report.model_size_kb > 0
    assert report.accuracy >= 0.0
    assert report.f1_macro >= 0.0
    assert report.precision_macro >= 0.0
    assert report.recall_macro >= 0.0
    assert len(report.classes) == 3
    assert len(report.confusion_matrix) == 3
    assert report.classifier_latency.mean_ms > 0
    assert report.cpu_info is not None

    report_dict = report.to_dict()
    assert "accuracy" in report_dict
    assert "confusion_matrix" in report_dict

    # Check print summary does not raise
    report.print_summary()


def test_random_forest_thread_enforcement(synthetic_classification_data):
    from src.evaluation.evaluate import evaluate_model

    X_train, y_train, X_test, y_test = synthetic_classification_data
    rf = RandomForestGestureClassifier(n_estimators=10, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    assert rf.clf.n_jobs == -1

    # Run evaluation with num_threads=1
    evaluate_model(
        model=rf,
        X_test=X_test,
        y_test=y_test,
        n_latency_iterations=10,
        benchmark_e2e=False,
        num_threads=1,
    )

    # Verify that original n_jobs was safely restored
    assert rf.clf.n_jobs == -1


def test_load_and_validate_model(synthetic_classification_data, tmp_path: Path):
    from src.evaluation.evaluate import load_and_validate_model

    X_train, y_train, _, _ = synthetic_classification_data
    rf = RandomForestGestureClassifier(n_estimators=5, random_state=42)
    rf.fit(X_train, y_train)
    rf_path = rf.save(tmp_path / "test_rf.joblib")

    mlp = MLPGestureClassifier(max_iter=10, random_state=42)
    mlp.fit(X_train, y_train)
    mlp_path = mlp.save(tmp_path / "test_mlp.joblib")

    # Correct types
    loaded_rf = load_and_validate_model(rf_path, expected_type="rf")
    assert isinstance(loaded_rf, RandomForestGestureClassifier)

    loaded_mlp = load_and_validate_model(mlp_path, expected_type="mlp")
    assert isinstance(loaded_mlp, MLPGestureClassifier)

    # Mismatched types should raise ValueError
    with pytest.raises(ValueError, match="Model type mismatch"):
        load_and_validate_model(rf_path, expected_type="mlp")

    with pytest.raises(ValueError, match="Model type mismatch"):
        load_and_validate_model(mlp_path, expected_type="rf")


def test_e2e_benchmark_rejects_missing_or_blank_image(synthetic_classification_data):
    from src.evaluation.evaluate import benchmark_end_to_end_latency, evaluate_model
    from src.landmarks.hand_detector import HandDetector
    from src.landmarks.normalizer import LandmarkNormalizer

    X_train, y_train, X_test, y_test = synthetic_classification_data
    rf = RandomForestGestureClassifier(n_estimators=5, random_state=42)
    rf.fit(X_train, y_train)

    # 1. Missing benchmark_image in evaluate_model
    with pytest.raises(ValueError, match="requires a valid hand image"):
        evaluate_model(
            model=rf,
            X_test=X_test,
            y_test=y_test,
            detector=HandDetector(),
            benchmark_image=None,
            benchmark_e2e=True,
        )

    # 2. Blank frame with 0 detected hands must be rejected
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="detected 0 hands"):
        benchmark_end_to_end_latency(
            detector=HandDetector(),
            normalizer=LandmarkNormalizer(),
            model=rf,
            benchmark_image=blank_frame,
            n_iterations=5,
        )



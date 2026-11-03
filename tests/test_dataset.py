"""Unit tests for dataset loading and preprocessing."""

import csv
from pathlib import Path
import tempfile
import numpy as np
import pytest

from src.data.dataset import DatasetLoader, DatasetSplits, SplitData
from src.landmarks.normalizer import LandmarkNormalizer


def test_split_data_validation():
    # Valid SplitData
    X = np.ones((5, 63), dtype=np.float32)
    y = np.array(["fist", "palm", "fist", "palm", "peace"])
    split = SplitData(X=X, y=y)
    assert split.num_samples == 5

    # Mismatched lengths
    with pytest.raises(ValueError, match="Mismatch between number of samples"):
        SplitData(X=np.ones((5, 63)), y=np.array(["fist", "palm"]))

    # Invalid feature count (e.g. 50 instead of 63)
    with pytest.raises(ValueError, match="Expected X to have shape"):
        SplitData(X=np.ones((5, 50)), y=y)


def test_load_from_arrays_preserves_splits_and_labels():
    loader = DatasetLoader(normalize=False)

    X_train = np.ones((10, 63), dtype=np.float32) * 1.0
    y_train = ["thumb_up"] * 5 + ["peace"] * 5

    X_val = np.ones((4, 63), dtype=np.float32) * 2.0
    y_val = ["thumb_up"] * 2 + ["peace"] * 2

    X_test = np.ones((6, 63), dtype=np.float32) * 3.0
    y_test = ["thumb_up"] * 3 + ["peace"] * 3

    splits = loader.load_from_arrays(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_val=X_val,
        y_val=y_val,
    )

    # Check split sizes
    assert splits.train.num_samples == 10
    assert splits.val is not None
    assert splits.val.num_samples == 4
    assert splits.test.num_samples == 6

    # Verify no data leakage or modification between splits
    assert np.all(splits.train.X == 1.0)
    assert np.all(splits.val.X == 2.0)
    assert np.all(splits.test.X == 3.0)

    # Check labels preserved
    assert list(splits.train.y) == y_train
    assert list(splits.val.y) == y_val
    assert list(splits.test.y) == y_test
    assert splits.classes == ["peace", "thumb_up"]


def test_load_from_arrays_with_normalization():
    loader = DatasetLoader(normalize=True)

    # Raw landmark arrays of shape (N, 21, 3)
    X_train_raw = np.zeros((4, 21, 3), dtype=np.float32)
    # Give landmark 12 a known offset
    X_train_raw[:, 12, :] = [2.0, 0.0, 0.0]
    X_train_raw[:, 5, :] = [1.0, 0.0, 0.0]
    y_train = ["fist", "palm", "fist", "palm"]

    X_test_raw = np.zeros((2, 21, 3), dtype=np.float32)
    X_test_raw[:, 12, :] = [4.0, 0.0, 0.0]
    X_test_raw[:, 5, :] = [2.0, 0.0, 0.0]
    y_test = ["fist", "palm"]

    splits = loader.load_from_arrays(
        X_train=X_train_raw,
        y_train=y_train,
        X_test=X_test_raw,
        y_test=y_test,
    )

    assert splits.is_normalized is True
    assert splits.train.X.shape == (4, 63)
    assert splits.test.X.shape == (2, 63)

    # After normalization by landmark 12 distance, landmark 12 should have norm ≈ 1.0
    for split in (splits.train, splits.test):
        reshaped = split.X.reshape(-1, 21, 3)
        np.testing.assert_allclose(np.linalg.norm(reshaped[:, 12, :], axis=-1), 1.0, atol=1e-5)
        # Landmark 0 should be 0.0
        np.testing.assert_allclose(reshaped[:, 0, :], 0.0, atol=1e-6)


def test_load_from_npz(tmp_path: Path):
    loader = DatasetLoader(normalize=False)

    npz_file = tmp_path / "dataset.npz"
    X_tr = np.zeros((8, 63), dtype=np.float32)
    y_tr = np.array(["ok", "fist"] * 4)
    X_te = np.zeros((4, 63), dtype=np.float32)
    y_te = np.array(["ok", "fist"] * 2)

    np.savez(npz_file, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te, classes=["fist", "ok"])

    splits = loader.load_from_npz(npz_file)
    assert splits.train.num_samples == 8
    assert splits.test.num_samples == 4
    assert splits.classes == ["fist", "ok"]


def test_load_from_npz_key_aliases(tmp_path: Path):
    loader = DatasetLoader(normalize=False)

    npz_file = tmp_path / "aliases.npz"
    X_tr = np.zeros((6, 63), dtype=np.float32)
    y_tr = np.array(["a", "b"] * 3)
    X_te = np.zeros((2, 63), dtype=np.float32)
    y_te = np.array(["a", "b"])

    # Use alias keys: train_x, train_y, test_x, test_y
    np.savez(npz_file, train_x=X_tr, train_y=y_tr, test_x=X_te, test_y=y_te)

    splits = loader.load_from_npz(npz_file)
    assert splits.train.num_samples == 6
    assert splits.test.num_samples == 2


def test_load_from_csv_with_metadata_columns(tmp_path: Path):
    loader = DatasetLoader(normalize=False)

    feat_cols = [f"lm_{i}" for i in range(63)]
    fieldnames = ["id", "subject_id", "label"] + feat_cols

    train_file = tmp_path / "train_meta.csv"
    with open(train_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(4):
            row = {"id": f"sample_{i}", "subject_id": "sub_1", "label": "peace"}
            row.update({c: 0.1 for c in feat_cols})
            writer.writerow(row)

    test_file = tmp_path / "test_meta.csv"
    with open(test_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(2):
            row = {"id": f"test_{i}", "subject_id": "sub_2", "label": "peace"}
            row.update({c: 0.2 for c in feat_cols})
            writer.writerow(row)

    splits = loader.load_from_csv_files(train_csv=train_file, test_csv=test_file)
    assert splits.train.num_samples == 4
    assert splits.test.num_samples == 2
    assert splits.train.X.shape == (4, 63)
    assert splits.test.X.shape == (2, 63)


def test_load_from_csv_files(tmp_path: Path):
    loader = DatasetLoader(normalize=False)

    feat_cols = [f"lm_{i}" for i in range(63)]
    fieldnames = ["label"] + feat_cols

    # Create train.csv
    train_file = tmp_path / "train.csv"
    with open(train_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(5):
            row = {"label": "fist" if i % 2 == 0 else "palm"}
            row.update({c: 0.1 * i for c in feat_cols})
            writer.writerow(row)

    # Create test.csv
    test_file = tmp_path / "test.csv"
    with open(test_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(3):
            row = {"label": "fist"}
            row.update({c: 0.5 for c in feat_cols})
            writer.writerow(row)

    splits = loader.load_from_csv_files(train_csv=train_file, test_csv=test_file)
    assert splits.train.num_samples == 5
    assert splits.test.num_samples == 3
    assert splits.train.X.shape == (5, 63)
    assert splits.test.X.shape == (3, 63)


def test_load_from_single_csv(tmp_path: Path):
    loader = DatasetLoader(normalize=False)

    feat_cols = [f"feat_{i}" for i in range(63)]
    fieldnames = ["split", "label"] + feat_cols

    csv_file = tmp_path / "dataset.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Train rows
        for _ in range(6):
            row = {"split": "train", "label": "thumbs_up"}
            row.update({c: 0.0 for c in feat_cols})
            writer.writerow(row)
        # Val rows
        for _ in range(2):
            row = {"split": "validation", "label": "thumbs_up"}
            row.update({c: 0.0 for c in feat_cols})
            writer.writerow(row)
        # Test rows
        for _ in range(4):
            row = {"split": "test", "label": "thumbs_up"}
            row.update({c: 0.0 for c in feat_cols})
            writer.writerow(row)

    splits = loader.load_from_single_csv(csv_file)
    assert splits.train.num_samples == 6
    assert splits.val is not None
    assert splits.val.num_samples == 2
    assert splits.test.num_samples == 4


def test_dataset_missing_raises_informative_error():
    loader = DatasetLoader()
    with pytest.raises(FileNotFoundError, match=r"\[DatasetPending\]"):
        loader.load_dataset("non_existent_dir_12345")

"""Dataset loading and preprocessing module for hand gesture landmarks.

This module is responsible for loading landmark samples, preserving official
train/validation/test splits provided by Person 1, and applying deterministic
coordinate normalization while strictly avoiding data leakage.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.landmarks.normalizer import LandmarkNormalizer


@dataclass
class SplitData:
    """Container for a single dataset split."""

    X: np.ndarray  # Shape: (N, 63), dtype: float32
    y: np.ndarray  # Shape: (N,), dtype: object or int
    sample_ids: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float32)
        self.y = np.asarray(self.y)
        if len(self.X) != len(self.y):
            raise ValueError(
                f"Mismatch between number of samples in X ({len(self.X)}) "
                f"and labels y ({len(self.y)})."
            )
        if self.X.ndim != 2 or self.X.shape[1] != 63:
            raise ValueError(
                f"Expected X to have shape (N, 63), got {self.X.shape}."
            )
        if self.sample_ids is not None and len(self.sample_ids) != len(self.X):
            raise ValueError(
                f"Mismatch between sample_ids ({len(self.sample_ids)}) "
                f"and samples ({len(self.X)})."
            )

    @property
    def num_samples(self) -> int:
        return len(self.X)


@dataclass
class DatasetSplits:
    """Container for complete dataset splits (train, validation, test)."""

    train: SplitData
    test: SplitData
    val: Optional[SplitData] = None
    classes: List[str] = field(default_factory=list)
    is_normalized: bool = False

    def __post_init__(self) -> None:
        if not self.classes:
            unique_labels = sorted(list(set(self.train.y.tolist())))
            self.classes = [str(c) for c in unique_labels]

    def summary(self) -> str:
        val_count = self.val.num_samples if self.val is not None else 0
        total = self.train.num_samples + val_count + self.test.num_samples
        lines = [
            f"Dataset Summary (Normalized: {self.is_normalized}):",
            f" - Total samples: {total}",
            f" - Train samples: {self.train.num_samples} ({self.train.num_samples/total*100:.1f}%)" if total else "",
            f" - Val samples:   {val_count} ({val_count/total*100:.1f}%)" if total else "",
            f" - Test samples:  {self.test.num_samples} ({self.test.num_samples/total*100:.1f}%)" if total else "",
            f" - Classes ({len(self.classes)}): {', '.join(self.classes)}",
        ]
        return "\n".join([line for line in lines if line])


DEFAULT_METADATA_COLUMNS = {
    "id",
    "sample_id",
    "subject",
    "subject_id",
    "participant",
    "handedness",
    "frame",
    "frame_idx",
    "timestamp",
    "date",
}


def _resolve_feature_columns(
    available_cols: Sequence[str],
    label_column: str,
    split_column: Optional[str] = None,
    feature_columns: Optional[Sequence[str]] = None,
    metadata_columns: Optional[Sequence[str]] = None,
) -> List[str]:
    """Identify and validate 63 feature columns from available CSV headers."""
    if feature_columns is not None:
        feat_list = list(feature_columns)
        if len(feat_list) != 63:
            raise ValueError(
                f"Configured feature_columns must contain exactly 63 columns, got {len(feat_list)}."
            )
        missing = [c for c in feat_list if c not in available_cols]
        if missing:
            raise KeyError(
                f"Specified feature columns not found in CSV: {missing[:5]}..."
            )
        return feat_list

    # Exclude reserved columns
    excluded = {label_column}
    if split_column:
        excluded.add(split_column)
    if metadata_columns:
        excluded.update(metadata_columns)
    excluded.update(DEFAULT_METADATA_COLUMNS)

    feat_cols = [c for c in available_cols if c not in excluded]

    if len(feat_cols) == 63:
        return feat_cols

    # If still not 63, check if non-excluded without default metadata equals 63
    non_reserved = [c for c in available_cols if c != label_column and c != split_column]
    if len(non_reserved) == 63:
        return non_reserved

    raise ValueError(
        f"Expected exactly 63 landmark feature columns, but identified {len(feat_cols)} "
        f"(from available columns: {list(available_cols)}). "
        f"Please provide 'feature_columns' or specify 'metadata_columns' to exclude."
    )


class DatasetLoader:
    """Configurable loader for hand gesture landmark datasets."""

    def __init__(
        self,
        normalizer: Optional[LandmarkNormalizer] = None,
        normalize: bool = True,
    ) -> None:
        """Initialize DatasetLoader.

        Args:
            normalizer: LandmarkNormalizer instance. If None, uses default.
            normalize: Whether to apply landmark normalization upon loading.
        """
        self.normalizer = normalizer or LandmarkNormalizer()
        self.normalize = normalize

    def load_from_arrays(
        self,
        X_train: np.ndarray,
        y_train: Sequence[Any],
        X_test: np.ndarray,
        y_test: Sequence[Any],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[Sequence[Any]] = None,
        classes: Optional[Sequence[str]] = None,
    ) -> DatasetSplits:
        """Create DatasetSplits directly from arrays or lists.

        Useful when Person 1 provides pre-loaded splits or for testing.
        """
        def _process_split(X_raw: np.ndarray, y_raw: Sequence[Any]) -> SplitData:
            X_arr = np.asarray(X_raw, dtype=np.float32)
            y_arr = np.asarray(y_raw)
            if self.normalize:
                # Normalize batch
                X_arr = self.normalizer.normalize_batch(X_arr)
            elif X_arr.ndim == 3 and X_arr.shape[1:] == (21, 3):
                X_arr = X_arr.reshape(-1, 63)
            return SplitData(X=X_arr, y=y_arr)

        train_split = _process_split(X_train, y_train)
        test_split = _process_split(X_test, y_test)
        val_split = None
        if X_val is not None and y_val is not None:
            val_split = _process_split(X_val, y_val)

        class_list = (
            [str(c) for c in classes]
            if classes is not None
            else sorted(list(set([str(y) for y in y_train])))
        )

        return DatasetSplits(
            train=train_split,
            val=val_split,
            test=test_split,
            classes=class_list,
            is_normalized=self.normalize,
        )

    def load_from_npz(
        self,
        npz_path: Union[str, Path],
    ) -> DatasetSplits:
        """Load dataset splits from a NumPy NPZ archive supporting standard key aliases."""
        path = Path(npz_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset NPZ file not found at: {path.resolve()}.\n"
                f"Please ensure Person 1's dataset archive is available."
            )

        data = np.load(path, allow_pickle=True)

        def _get_key(candidates: Sequence[str]) -> Optional[str]:
            for k in candidates:
                if k in data:
                    return k
            return None

        k_xtr = _get_key(["X_train", "x_train", "train_x", "train_X", "X_tr", "x_tr"])
        k_ytr = _get_key(["y_train", "Y_train", "train_y", "train_Y", "y_tr", "Y_tr"])
        k_xte = _get_key(["X_test", "x_test", "test_x", "test_X", "X_te", "x_te"])
        k_yte = _get_key(["y_test", "Y_test", "test_y", "test_Y", "y_te", "Y_te"])

        if not k_xtr or not k_ytr:
            raise KeyError(
                f"NPZ archive at {path} must contain training features and labels (e.g. 'X_train', 'y_train'). "
                f"Keys found: {list(data.keys())}"
            )
        if not k_xte or not k_yte:
            raise KeyError(
                f"NPZ archive at {path} must contain test features and labels (e.g. 'X_test', 'y_test'). "
                f"Keys found: {list(data.keys())}"
            )

        X_train = data[k_xtr]
        y_train = data[k_ytr]
        X_test = data[k_xte]
        y_test = data[k_yte]

        k_xval = _get_key(["X_val", "x_val", "val_x", "val_X", "X_valid", "x_valid", "valid_x"])
        k_yval = _get_key(["y_val", "Y_val", "val_y", "val_Y", "y_valid", "Y_valid", "valid_y"])
        X_val = data[k_xval] if k_xval else None
        y_val = data[k_yval] if k_yval else None

        k_classes = _get_key(["classes", "class_names", "labels", "label_names", "gesture_classes"])
        classes = data[k_classes] if k_classes else None
        if classes is not None:
            classes = [str(c) for c in classes]

        return self.load_from_arrays(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val,
            classes=classes,
        )

    def load_from_csv_files(
        self,
        train_csv: Union[str, Path],
        test_csv: Union[str, Path],
        val_csv: Optional[Union[str, Path]] = None,
        label_column: str = "label",
        feature_columns: Optional[Sequence[str]] = None,
        metadata_columns: Optional[Sequence[str]] = None,
    ) -> DatasetSplits:
        """Load pre-split CSV files (train.csv, test.csv, optional val.csv)."""
        train_path = Path(train_csv)
        test_path = Path(test_csv)

        if not train_path.is_file():
            raise FileNotFoundError(f"Train CSV not found: {train_path.resolve()}")
        if not test_path.is_file():
            raise FileNotFoundError(f"Test CSV not found: {test_path.resolve()}")

        def _read_csv(p: Path) -> Tuple[np.ndarray, np.ndarray]:
            rows_X = []
            rows_y = []
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError(f"CSV file {p} is empty or invalid.")

                cols = list(reader.fieldnames)
                if label_column not in cols:
                    raise KeyError(
                        f"Label column '{label_column}' not found in {p}. "
                        f"Available columns: {cols}"
                    )

                feat_cols = _resolve_feature_columns(
                    available_cols=cols,
                    label_column=label_column,
                    feature_columns=feature_columns,
                    metadata_columns=metadata_columns,
                )

                for row in reader:
                    feat_vals = [float(row[c]) for c in feat_cols]
                    rows_X.append(feat_vals)
                    rows_y.append(row[label_column])

            if not rows_X:
                raise ValueError(f"No data rows found in {p}.")

            return np.array(rows_X, dtype=np.float32), np.array(rows_y)

        X_train, y_train = _read_csv(train_path)
        X_test, y_test = _read_csv(test_path)

        X_val, y_val = None, None
        if val_csv is not None:
            val_path = Path(val_csv)
            if val_path.is_file():
                X_val, y_val = _read_csv(val_path)

        return self.load_from_arrays(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val,
        )

    def load_from_single_csv(
        self,
        csv_path: Union[str, Path],
        split_column: str = "split",
        label_column: str = "label",
        feature_columns: Optional[Sequence[str]] = None,
        metadata_columns: Optional[Sequence[str]] = None,
    ) -> DatasetSplits:
        """Load dataset from a single CSV containing a split column."""
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset CSV not found at: {path.resolve()}")

        splits_data: Dict[str, Tuple[List[List[float]], List[str]]] = {
            "train": ([], []),
            "val": ([], []),
            "test": ([], []),
        }

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV {path} is empty.")

            cols = list(reader.fieldnames)
            for required in (split_column, label_column):
                if required not in cols:
                    raise KeyError(
                        f"Required column '{required}' not found in CSV. "
                        f"Columns: {cols}"
                    )

            feat_cols = _resolve_feature_columns(
                available_cols=cols,
                label_column=label_column,
                split_column=split_column,
                feature_columns=feature_columns,
                metadata_columns=metadata_columns,
            )

            for row in reader:
                s = row[split_column].strip().lower()
                # Map common split alias names
                if s in ("validation", "valid", "val"):
                    s = "val"
                elif s in ("train", "training"):
                    s = "train"
                elif s in ("test", "testing"):
                    s = "test"
                else:
                    raise ValueError(
                        f"Unknown split identifier '{row[split_column]}' in row."
                    )

                feat_vals = [float(row[c]) for c in feat_cols]
                splits_data[s][0].append(feat_vals)
                splits_data[s][1].append(row[label_column])

        if not splits_data["train"][0]:
            raise ValueError(f"No train samples found in {path}.")
        if not splits_data["test"][0]:
            raise ValueError(f"No test samples found in {path}.")

        X_train = np.array(splits_data["train"][0], dtype=np.float32)
        y_train = np.array(splits_data["train"][1])

        X_test = np.array(splits_data["test"][0], dtype=np.float32)
        y_test = np.array(splits_data["test"][1])

        X_val, y_val = None, None
        if splits_data["val"][0]:
            X_val = np.array(splits_data["val"][0], dtype=np.float32)
            y_val = np.array(splits_data["val"][1])

        return self.load_from_arrays(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val,
        )

    def load_dataset(
        self,
        data_dir: Union[str, Path] = "data",
    ) -> DatasetSplits:
        """Autodetect and load the dataset from the data directory.

        Checks for standard files in order:
        1. npz: data_dir/dataset.npz or data_dir/landmarks.npz
        2. separate csv: data_dir/train.csv and data_dir/test.csv
        3. single csv: data_dir/dataset.csv or data_dir/gestures.csv

        Raises:
            FileNotFoundError: If no official dataset file from Person 1 is found.
        """
        dir_path = Path(data_dir)
        if not dir_path.exists():
            raise FileNotFoundError(
                f"[DatasetPending] Dataset directory '{dir_path.resolve()}' does not exist.\n"
                f"The common dataset and split from Person 1 have not been imported yet.\n"
                f"To provide the dataset, place one of the following in '{dir_path}':\n"
                f"  1. 'dataset.npz' containing (X_train, y_train, X_test, y_test, [X_val, y_val])\n"
                f"  2. 'train.csv', 'test.csv' (and optional 'val.csv')\n"
                f"  3. 'dataset.csv' with a 'split' column ('train', 'val', 'test') and 63 landmark columns."
            )

        # 1. Check NPZ files
        for name in ("dataset.npz", "landmarks.npz", "hand_gestures.npz"):
            npz_candidate = dir_path / name
            if npz_candidate.is_file():
                print(f"[DatasetLoader] Found NPZ dataset at: {npz_candidate}")
                return self.load_from_npz(npz_candidate)

        # 2. Check split CSV files
        train_csv = dir_path / "train.csv"
        test_csv = dir_path / "test.csv"
        val_csv = dir_path / "val.csv"
        if not val_csv.is_file():
            val_csv = dir_path / "validation.csv"
        if train_csv.is_file() and test_csv.is_file():
            print(f"[DatasetLoader] Found split CSVs in {dir_path}")
            return self.load_from_csv_files(
                train_csv=train_csv,
                test_csv=test_csv,
                val_csv=val_csv if val_csv.is_file() else None,
            )

        # 3. Check single CSV files
        for name in ("dataset.csv", "gestures.csv", "landmarks.csv"):
            single_csv = dir_path / name
            if single_csv.is_file():
                print(f"[DatasetLoader] Found single dataset CSV at: {single_csv}")
                return self.load_from_single_csv(single_csv)

        raise FileNotFoundError(
            f"[DatasetPending] No recognized dataset files found in '{dir_path.resolve()}'.\n"
            f"Expected Person 1's dataset in one of these formats:\n"
            f"  - {dir_path}/dataset.npz (keys: X_train, y_train, X_test, y_test, [X_val, y_val])\n"
            f"  - {dir_path}/train.csv and {dir_path}/test.csv\n"
            f"  - {dir_path}/dataset.csv with 'split' column"
        )

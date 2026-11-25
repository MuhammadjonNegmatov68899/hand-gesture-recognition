from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

DEFAULT_IMG_SIZE = 224


def read_splits(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_bgr(frame_bgr: np.ndarray, img_size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)


def load_image(path: str | Path, img_size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Cannot read image: {path}")
    return preprocess_bgr(frame, img_size)


def load_split_arrays(
    splits: Dict, split: str, root: str | Path, img_size: int = DEFAULT_IMG_SIZE
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    classes = splits["classes"]
    index = {c: i for i, c in enumerate(classes)}
    items = splits.get(split, [])
    X = np.zeros((len(items), img_size, img_size, 3), dtype=np.uint8)
    y = np.zeros((len(items),), dtype=np.int64)
    ids = []
    for i, item in enumerate(items):
        X[i] = load_image(Path(root) / item["path"], img_size)
        y[i] = index[item["label"]]
        ids.append(item["path"])
    return X, y, ids

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from src.landmarks.hand_detector import DEFAULT_MODEL_PATH, DEFAULT_MODEL_URL

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def ensure_model(path: Path) -> Path:
    if not path.is_file():
        print(f"Downloading hand landmarker model to {path} ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DEFAULT_MODEL_URL, str(path))
    return path


def extract_landmarks(landmarker, image_path: Path) -> Optional[np.ndarray]:
    data = np.fromfile(str(image_path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.hand_landmarks:
        return None
    return np.array([(p.x, p.y, p.z) for p in result.hand_landmarks[0]], dtype=np.float32)


def subject_of(path: Path) -> str:
    return path.stem.split("_")[0]


def assign_splits(
    samples: List[Dict], val_ratio: float, test_ratio: float, seed: int,
    val_subjects: List[str], test_subjects: List[str],
) -> Dict[str, List[Dict]]:
    out = {"train": [], "val": [], "test": []}
    if val_subjects or test_subjects:
        for s in samples:
            if s["subject"] in test_subjects:
                out["test"].append(s)
            elif s["subject"] in val_subjects:
                out["val"].append(s)
            else:
                out["train"].append(s)
        return out
    rng = random.Random(seed)
    by_class: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        by_class[s["label"]].append(s)
    for label in sorted(by_class):
        items = sorted(by_class[label], key=lambda s: s["path"])
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        out["test"] += items[:n_test]
        out["val"] += items[n_test:n_test + n_val]
        out["train"] += items[n_test + n_val:]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create shared CNN/landmark dataset splits.")
    parser.add_argument("--images-dir", default="data/images")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--val-subjects", default="", help="Comma-separated subject ids for validation.")
    parser.add_argument("--test-subjects", default="", help="Comma-separated subject ids for test.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    data_dir = Path(args.data_dir)
    classes = sorted(d.name for d in images_dir.iterdir() if d.is_dir())
    if len(classes) < 2:
        raise SystemExit(f"Need at least 2 class folders in {images_dir}")

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model(Path(args.model_path)))),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )

    samples: List[Dict] = []
    dropped = defaultdict(int)
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for label in classes:
            files = sorted(p for p in (images_dir / label).iterdir() if p.suffix.lower() in IMAGE_EXTS)
            for i, path in enumerate(files, 1):
                lm = extract_landmarks(landmarker, path)
                if lm is None:
                    dropped[label] += 1
                    continue
                samples.append({
                    "path": path.relative_to(data_dir).as_posix() if path.is_relative_to(data_dir) else path.as_posix(),
                    "label": label,
                    "subject": subject_of(path),
                    "landmarks": lm,
                })
                if i % 100 == 0:
                    print(f"  {label}: {i}/{len(files)}")

    val_subjects = [s for s in args.val_subjects.split(",") if s]
    test_subjects = [s for s in args.test_subjects.split(",") if s]
    splits = assign_splits(samples, args.val_ratio, args.test_ratio, args.seed, val_subjects, test_subjects)

    data_dir.mkdir(parents=True, exist_ok=True)
    split_json = {"classes": classes, "root": data_dir.as_posix()}
    npz = {"classes": np.array(classes)}
    for name, items in splits.items():
        split_json[name] = [{"path": s["path"], "label": s["label"], "subject": s["subject"]} for s in items]
        npz[f"X_{name}"] = np.stack([s["landmarks"] for s in items]) if items else np.zeros((0, 21, 3), np.float32)
        npz[f"y_{name}"] = np.array([s["label"] for s in items])
        npz[f"ids_{name}"] = np.array([s["path"] for s in items])

    with open(data_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(split_json, f, indent=2)
    np.savez_compressed(data_dir / "dataset.npz", **npz)

    print(f"Classes: {classes}")
    for name in ("train", "val", "test"):
        counts = defaultdict(int)
        for s in splits[name]:
            counts[s["label"]] += 1
        print(f"{name:>5}: {len(splits[name])} samples {dict(counts)}")
    if dropped:
        print(f"Dropped (no hand detected): {dict(dropped)}")
    print(f"Saved {data_dir / 'splits.json'} and {data_dir / 'dataset.npz'}")


if __name__ == "__main__":
    main()

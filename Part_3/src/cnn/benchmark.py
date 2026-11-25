from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import psutil
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.cnn.common import preprocess_bgr, read_splits


def make_interpreter(path: Path, threads: int):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    interpreter = Interpreter(model_path=str(path), num_threads=threads)
    interpreter.allocate_tensors()
    return interpreter


class TFLiteClassifier:
    def __init__(self, path: Path, threads: int) -> None:
        self.it = make_interpreter(path, threads)
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.img_size = int(self.inp["shape"][1])
        self.in_dtype = self.inp["dtype"]
        self.in_scale, self.in_zp = self.inp["quantization"]
        self.out_scale, self.out_zp = self.out["quantization"]

    def prepare(self, rgb: np.ndarray) -> np.ndarray:
        if self.in_dtype == np.float32:
            return rgb.astype(np.float32)[None, ...]
        info = np.iinfo(self.in_dtype)
        q = np.round(rgb.astype(np.float32) / self.in_scale + self.in_zp)
        return np.clip(q, info.min, info.max).astype(self.in_dtype)[None, ...]

    def classify(self, x: np.ndarray) -> np.ndarray:
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        y = self.it.get_tensor(self.out["index"])[0]
        if self.out["dtype"] != np.float32:
            y = (y.astype(np.float32) - self.out_zp) * self.out_scale
        return y


def stats(values_ms: List[float]) -> Dict[str, float]:
    a = np.asarray(values_ms, dtype=np.float64)
    if a.size == 0:
        return {}
    mean = float(a.mean())
    return {
        "mean_ms": mean,
        "std_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "fps": 1000.0 / mean if mean > 0 else 0.0,
        "iterations": int(a.size),
    }


def read_bgr(path: Path) -> np.ndarray:
    frame = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Cannot read image: {path}")
    return frame


class ResourceMonitor:
    def __init__(self) -> None:
        self.proc = psutil.Process()
        self.peak_rss = 0

    def start(self) -> None:
        self.t0 = time.perf_counter()
        self.c0 = self.proc.cpu_times()
        self.sample()

    def sample(self) -> None:
        self.peak_rss = max(self.peak_rss, self.proc.memory_info().rss)

    def stop(self) -> Dict[str, float]:
        wall = time.perf_counter() - self.t0
        c1 = self.proc.cpu_times()
        busy = (c1.user - self.c0.user) + (c1.system - self.c0.system)
        cpu_pct = 100.0 * busy / wall if wall > 0 else 0.0
        return {
            "cpu_percent_one_core": cpu_pct,
            "cpu_percent_all_cores": cpu_pct / (psutil.cpu_count() or 1),
            "peak_rss_mb": self.peak_rss / (1024 * 1024),
            "wall_time_s": wall,
        }


def evaluate_accuracy(clf: TFLiteClassifier, items: List[Dict], root: Path, classes: List[str]) -> Dict:
    y_true, y_pred = [], []
    index = {c: i for i, c in enumerate(classes)}
    for item in items:
        rgb = preprocess_bgr(read_bgr(root / item["path"]), clf.img_size)
        y_pred.append(int(np.argmax(clf.classify(clf.prepare(rgb)))))
        y_true.append(index[item["label"]])
    labels = list(range(len(classes)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "test_samples": len(y_true),
    }


def benchmark_offline(clf: TFLiteClassifier, frames: List[np.ndarray], warmup: int, iterations: int) -> Dict:
    for i in range(warmup):
        clf.classify(clf.prepare(preprocess_bgr(frames[i % len(frames)], clf.img_size)))
    t_ext, t_clf, t_tot = [], [], []
    mon = ResourceMonitor()
    mon.start()
    for i in range(iterations):
        frame = frames[i % len(frames)]
        t0 = time.perf_counter()
        x = clf.prepare(preprocess_bgr(frame, clf.img_size))
        t1 = time.perf_counter()
        clf.classify(x)
        t2 = time.perf_counter()
        t_ext.append((t1 - t0) * 1000)
        t_clf.append((t2 - t1) * 1000)
        t_tot.append((t2 - t0) * 1000)
        if i % 50 == 0:
            mon.sample()
    return {"T_extract": stats(t_ext), "T_classify": stats(t_clf), "T_total": stats(t_tot), "resources": mon.stop()}


def benchmark_camera(clf: TFLiteClassifier, camera: int, warmup: int, iterations: int) -> Dict:
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera}")
    try:
        for _ in range(warmup):
            ok, frame = cap.read()
            if ok:
                clf.classify(clf.prepare(preprocess_bgr(frame, clf.img_size)))
        t_cap, t_ext, t_clf, t_tot = [], [], [], []
        mon = ResourceMonitor()
        mon.start()
        n = 0
        while n < iterations:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t1 = time.perf_counter()
            if not ok:
                continue
            x = clf.prepare(preprocess_bgr(frame, clf.img_size))
            t2 = time.perf_counter()
            clf.classify(x)
            t3 = time.perf_counter()
            t_cap.append((t1 - t0) * 1000)
            t_ext.append((t2 - t1) * 1000)
            t_clf.append((t3 - t2) * 1000)
            t_tot.append((t3 - t0) * 1000)
            n += 1
            if n % 50 == 0:
                mon.sample()
        res = mon.stop()
        return {
            "T_capture": stats(t_cap), "T_extract": stats(t_ext), "T_classify": stats(t_clf),
            "T_total": stats(t_tot), "sustained_fps": n / res["wall_time_s"], "resources": res,
        }
    finally:
        cap.release()


def fmt(s: Optional[Dict]) -> str:
    return f"{s['mean_ms']:8.3f} +- {s['std_ms']:6.3f}" if s else " " * 18


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only benchmark of CNN TFLite variants.")
    parser.add_argument("--models-dir", default="models/cnn")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--variants", default="fp32,fp16,int8")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--pin-core", type=int, default=None, help="Pin the process to one CPU core.")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--timing-frames", type=int, default=200, help="Test frames cached in RAM for offline timing.")
    parser.add_argument("--camera", type=int, default=None, help="Also run live end-to-end timing on this camera.")
    parser.add_argument("--output-json", default="results/cnn_benchmark.json")
    args = parser.parse_args()

    if args.pin_core is not None:
        psutil.Process().cpu_affinity([args.pin_core])

    splits = read_splits(args.splits)
    root = Path(splits.get("root", str(Path(args.splits).parent)))
    classes = splits["classes"]
    test_items = splits["test"]
    if not test_items:
        raise SystemExit("Test split is empty.")
    timing_frames = [read_bgr(root / it["path"]) for it in test_items[: args.timing_frames]]

    meta_path = Path(args.models_dir) / "train_meta.json"
    train_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    report = {
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": psutil.cpu_count(),
            "physical_cpus": psutil.cpu_count(logical=False),
            "threads": args.threads,
            "pinned_core": args.pin_core,
            "execution": "CPU only (TFLite interpreter, GPU disabled)",
        },
        "classes": classes,
        "training_time_sec": train_meta.get("training_time_sec"),
        "params": train_meta.get("params"),
        "variants": {},
    }

    for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
        path = Path(args.models_dir) / f"cnn_{name}.tflite"
        if not path.is_file():
            print(f"Skipping {name}: {path} not found")
            continue
        print(f"\n=== {name.upper()} ({path}) ===")
        clf = TFLiteClassifier(path, args.threads)
        entry = {"model_path": str(path), "model_size_kb": path.stat().st_size / 1024}
        entry.update(evaluate_accuracy(clf, test_items, root, classes))
        entry["offline"] = benchmark_offline(clf, timing_frames, args.warmup, args.iterations)
        if args.camera is not None:
            entry["camera"] = benchmark_camera(clf, args.camera, args.warmup, args.iterations)
        report["variants"][name] = entry
        off = entry["offline"]
        print(f"Accuracy {entry['accuracy'] * 100:.2f}% | F1 {entry['f1_macro']:.4f} | size {entry['model_size_kb']:.1f} KB")
        print(f"T_extract {fmt(off['T_extract'])} ms | T_classify {fmt(off['T_classify'])} ms | "
              f"{off['T_total']['fps']:.1f} FPS | RAM {off['resources']['peak_rss_mb']:.0f} MB | "
              f"CPU {off['resources']['cpu_percent_one_core']:.0f}%")

    print("\n" + "=" * 112)
    print(f"{'Variant':<8}{'Size KB':>10}{'Acc %':>8}{'F1':>8}{'T_extract ms':>22}{'T_classify ms':>22}{'FPS':>8}{'RAM MB':>9}{'CPU %':>8}")
    print("-" * 112)
    for name, e in report["variants"].items():
        o = e["offline"]
        print(f"{name:<8}{e['model_size_kb']:>10.1f}{e['accuracy'] * 100:>8.2f}{e['f1_macro']:>8.4f}"
              f"{fmt(o['T_extract']):>22}{fmt(o['T_classify']):>22}{o['T_total']['fps']:>8.1f}"
              f"{o['resources']['peak_rss_mb']:>9.0f}{o['resources']['cpu_percent_one_core']:>8.0f}")
        if "camera" in e:
            c = e["camera"]
            print(f"{'  live':<8} capture {fmt(c['T_capture'])} ms | total {fmt(c['T_total'])} ms | sustained {c['sustained_fps']:.1f} FPS")
    print("=" * 112)

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

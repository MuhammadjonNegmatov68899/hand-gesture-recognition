from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.cnn.common import load_split_arrays, read_splits


def converter_from(model):
    tmp = Path(tempfile.mkdtemp()) / "saved_model"
    model.export(str(tmp), format="tf_saved_model", verbose=False)
    return tf.lite.TFLiteConverter.from_saved_model(str(tmp))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CNN to FP32/FP16/INT8 TFLite.")
    parser.add_argument("--model", default="models/cnn/cnn_fp32.keras")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--output-dir", default="models/cnn")
    parser.add_argument("--calibration-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = keras.models.load_model(args.model)
    img_size = int(model.input_shape[1])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    splits = read_splits(args.splits)
    root = splits.get("root", str(Path(args.splits).parent))
    X_train, _, _ = load_split_arrays(splits, "train", root, img_size)
    rng = np.random.default_rng(args.seed)
    calib = X_train[rng.permutation(len(X_train))[: args.calibration_samples]].astype(np.float32)
    print(f"Calibration images for INT8: {len(calib)} (from train split only)")

    def representative_dataset():
        for img in calib:
            yield [img[None, ...]]

    results = {}

    conv = converter_from(model)
    results["fp32"] = conv.convert()

    conv = converter_from(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    results["fp16"] = conv.convert()

    conv = converter_from(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.uint8
    results["int8"] = conv.convert()

    sizes = {}
    for name, blob in results.items():
        path = out / f"cnn_{name}.tflite"
        path.write_bytes(blob)
        sizes[name] = round(len(blob) / 1024, 2)
        print(f"{name.upper():>4}: {path} ({sizes[name]:.1f} KB)")

    with open(out / "export_meta.json", "w", encoding="utf-8") as f:
        json.dump({"img_size": img_size, "calibration_samples": len(calib), "size_kb": sizes}, f, indent=2)


if __name__ == "__main__":
    main()

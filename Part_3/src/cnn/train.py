from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.cnn.common import DEFAULT_IMG_SIZE, load_split_arrays, read_splits


def build_model(num_classes: int, img_size: int, alpha: float, pretrained: bool, dropout: float):
    inputs = keras.Input(shape=(img_size, img_size, 3), dtype="float32", name="image")
    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="to_minus1_1")(inputs)
    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        alpha=alpha,
        include_top=False,
        weights="imagenet" if pretrained else None,
    )
    if pretrained:
        base.trainable = False
        x = base(x, training=False)
    else:
        for layer in base.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.momentum = 0.9
        x = base(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs, name="mobilenetv2_gesture"), base


def augment(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.random_brightness(image, 32.0)
    image = tf.image.random_contrast(image, 0.75, 1.25)
    size = tf.shape(image)[0]
    scale = tf.random.uniform([], 0.85, 1.0)
    crop = tf.cast(tf.cast(size, tf.float32) * scale, tf.int32)
    image = tf.image.random_crop(image, tf.stack([crop, crop, 3]))
    image = tf.image.resize(image, tf.stack([size, size]))
    return tf.clip_by_value(image, 0.0, 255.0), label


def to_float(image, label):
    return tf.cast(image, tf.float32), label


def make_ds(X, y, batch, training, seed):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(to_float, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 gesture CNN.")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--output-dir", default="models/cnn")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--alpha", type=float, default=1.0, help="MobileNetV2 width multiplier (0.35, 0.5, 0.75, 1.0).")
    parser.add_argument("--no-pretrained", action="store_true", help="Train from scratch instead of ImageNet weights.")
    parser.add_argument("--epochs-head", type=int, default=10)
    parser.add_argument("--epochs-finetune", type=int, default=10)
    parser.add_argument("--finetune-layers", type=int, default=40, help="Number of top base layers to unfreeze.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience on val_accuracy.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    keras.utils.set_random_seed(args.seed)
    splits = read_splits(args.splits)
    root = splits.get("root", str(Path(args.splits).parent))
    classes = splits["classes"]

    print("Loading images...")
    X_train, y_train, _ = load_split_arrays(splits, "train", root, args.img_size)
    X_val, y_val, _ = load_split_arrays(splits, "val", root, args.img_size)
    X_test, y_test, _ = load_split_arrays(splits, "test", root, args.img_size)
    if len(X_val) == 0:
        raise SystemExit("Validation split is empty.")
    print(f"train={len(X_train)} val={len(X_val)} test={len(X_test)} classes={classes}")

    train_ds = make_ds(X_train, y_train, args.batch_size, True, args.seed)
    val_ds = make_ds(X_val, y_val, args.batch_size, False, args.seed)
    test_ds = make_ds(X_test, y_test, args.batch_size, False, args.seed)

    model, base = build_model(len(classes), args.img_size, args.alpha, not args.no_pretrained, args.dropout)
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=args.patience, restore_best_weights=True)]

    t0 = time.perf_counter()
    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head, callbacks=callbacks, verbose=2)

    history = {k: list(map(float, v)) for k, v in h1.history.items()}
    if args.epochs_finetune > 0 and not args.no_pretrained:
        base.trainable = True
        for layer in base.layers[:-args.finetune_layers]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(optimizer=keras.optimizers.Adam(args.lr_finetune), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_finetune, callbacks=callbacks, verbose=2)
        for k, v in h2.history.items():
            history.setdefault(k, []).extend(map(float, v))
    training_time = time.perf_counter() - t0

    test_loss, test_acc = model.evaluate(test_ds, verbose=0) if len(X_test) else (float("nan"), float("nan"))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "cnn_fp32.keras")
    meta = {
        "classes": classes,
        "img_size": args.img_size,
        "alpha": args.alpha,
        "pretrained": not args.no_pretrained,
        "params": int(model.count_params()),
        "training_time_sec": training_time,
        "test_accuracy_keras": float(test_acc),
        "samples": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "args": vars(args),
    }
    with open(out / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(out / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, key in zip(axes, ("accuracy", "loss")):
            ax.plot(history.get(key, []), label="train")
            ax.plot(history.get(f"val_{key}", []), label="val")
            if args.epochs_finetune > 0 and not args.no_pretrained:
                ax.axvline(len(h1.history["loss"]) - 0.5, color="gray", linestyle="--", label="fine-tune start")
            ax.set_title(key)
            ax.set_xlabel("epoch")
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / "training_curves.png", dpi=120)
    except ImportError:
        print("matplotlib not installed: skipping training_curves.png")

    print(f"Training time: {training_time:.1f} s | params: {meta['params']:,} | test accuracy: {test_acc * 100:.2f}%")
    print(f"Saved model to {out / 'cnn_fp32.keras'}")


if __name__ == "__main__":
    main()

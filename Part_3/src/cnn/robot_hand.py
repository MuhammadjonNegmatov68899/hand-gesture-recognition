from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import serial
import serial.tools.list_ports

from src.cnn.benchmark import TFLiteClassifier
from src.cnn.common import preprocess_bgr, read_splits

OPEN, CLOSED = 180, 0

GESTURE_TO_FINGERS = {
    "fist":  [CLOSED, CLOSED, CLOSED, CLOSED, CLOSED],
    "palm":  [OPEN, OPEN, OPEN, OPEN, OPEN],
    "point": [CLOSED, OPEN, CLOSED, CLOSED, CLOSED],
    "peace": [CLOSED, OPEN, OPEN, CLOSED, CLOSED],
    "ok":    [CLOSED, CLOSED, OPEN, OPEN, OPEN],
}


def find_arduino_port():
    keys = ("arduino", "ch340", "ch341", "usb-serial", "usb serial", "cp210", "ftdi")
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        text = f"{p.device} {p.description} {p.manufacturer or ''}".lower()
        if any(k in text for k in keys) or p.vid in (0x2341, 0x2A03, 0x1A86, 0x10C4, 0x0403):
            return p.device
    return ports[0].device if len(ports) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Control the robot hand with CNN gesture predictions.")
    parser.add_argument("--model", default="models/cnn/cnn_int8.tflite")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--port", default=None)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--smooth", type=int, default=7, help="Majority vote over the last N predictions.")
    args = parser.parse_args()

    classes = read_splits(args.splits)["classes"]
    clf = TFLiteClassifier(Path(args.model), args.threads)

    port = args.port or find_arduino_port()
    ser = None
    if port:
        try:
            ser = serial.Serial(port, 115200, timeout=0)
            time.sleep(2.0)
            print(f"Arduino port: {port}")
        except Exception as e:
            print(f"Failed to open {port}: {e}. Running without the hand.")
    else:
        print("Arduino not found. Running without the hand.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    history = deque(maxlen=args.smooth)
    sent = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            probs = clf.classify(clf.prepare(preprocess_bgr(frame, clf.img_size)))
            ms = (time.perf_counter() - t0) * 1000
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            history.append(classes[idx] if conf >= args.min_confidence else None)

            votes = Counter(g for g in history if g)
            stable = None
            if votes:
                g, n = votes.most_common(1)[0]
                if n > len(history) // 2:
                    stable = g

            if stable and stable != sent and stable in GESTURE_TO_FINGERS and ser:
                ser.write(("S," + ",".join(map(str, GESTURE_TO_FINGERS[stable])) + "\n").encode("ascii"))
                sent = stable

            view = cv2.flip(frame, 1)
            cv2.putText(view, f"CNN: {classes[idx]} {conf * 100:.0f}%  ({ms:.1f} ms)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if conf >= args.min_confidence else (0, 165, 255), 2)
            cv2.putText(view, f"Hand: {sent or '-'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("CNN -> Robot hand", view)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ser and ser.is_open:
            ser.close()


if __name__ == "__main__":
    main()

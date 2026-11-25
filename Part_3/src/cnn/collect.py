from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture webcam frames per gesture class.")
    parser.add_argument("--classes", required=True, help="Comma-separated class names.")
    parser.add_argument("--out-dir", default="data/images")
    parser.add_argument("--subject", default="s1", help="Subject/session id used in file names.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.15, help="Seconds between saved frames.")
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not 1 <= len(classes) <= 9:
        raise SystemExit("Provide between 1 and 9 classes.")
    out_dir = Path(args.out_dir)
    for c in classes:
        (out_dir / c).mkdir(parents=True, exist_ok=True)
    counts = {c: len(list((out_dir / c).glob("*.jpg"))) for c in classes}

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    print("Press 1..9 to start/stop recording a class, q to quit:")
    for i, c in enumerate(classes, 1):
        print(f"  {i}: {c}")

    active = None
    last_saved = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        if active is not None and now - last_saved >= args.interval:
            name = f"{args.subject}_{int(now * 1000)}.jpg"
            ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok_enc:
                buf.tofile(str(out_dir / active / name))
                counts[active] += 1
            last_saved = now

        view = cv2.flip(frame, 1)
        status = f"REC: {active} ({counts[active]})" if active else "IDLE"
        cv2.putText(view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if active else (0, 200, 0), 2)
        for i, c in enumerate(classes, 1):
            cv2.putText(view, f"{i}: {c} = {counts[c]}", (10, 35 + 25 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Gesture dataset collection", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if ord("1") <= key < ord("1") + len(classes):
            chosen = classes[key - ord("1")]
            active = None if active == chosen else chosen

    cap.release()
    cv2.destroyAllWindows()
    print("Images per class:", counts)


if __name__ == "__main__":
    main()

"""Real-Time Hand Landmark Detector using MediaPipe Tasks Hand Landmarker.

Part of the university Hand Gesture Recognition project.
Captures live webcam video with OpenCV, processes frames through Google's
MediaPipe Tasks Vision HandLandmarker API, visualizes the 21 3D hand landmarks
and skeletal connections, and extracts normalized coordinates ready for downstream
gesture classification.

Model Information:
------------------
This script requires the MediaPipe Hand Landmarker task bundle (`hand_landmarker.task`).
The pretrained model detects hand bounding boxes and predicts 21 3D hand landmark
coordinates in normalized image space and world coordinates.

Download URL:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

Handedness Convention Note:
---------------------------
MediaPipe's Hand Landmarker model was trained on standard (unmirrored) photos from an
observer's perspective. In interactive webcam applications (selfie mode), frames are
horizontally flipped (`cv2.flip(frame, 1)`) so the display acts like a mirror.
Mirroring a 2D image inverts the 3D chirality of the hand (a physical right hand has
the geometry of an anatomical left hand when reflected). Consequently, on mirrored frames,
MediaPipe's raw prediction is inverted ("Left" for physical right hand). This detector
tracks whether the frame was mirrored and restores the correct physical handedness label.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
from typing import List, Optional, Tuple
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Tasks Vision imports (current supported API in mediapipe >= 0.10.x / 1.x)
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Default model download location and URL
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path("models/hand_landmarker.task")

# Names of the 21 hand landmarks defined by MediaPipe
HAND_LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]


@dataclass
class HandLandmarkData:
    """Structured container for hand landmarks extracted from a single hand."""

    handedness: str  # Physical hand: "Left" or "Right" (mirror-corrected if applicable)
    raw_handedness: str  # Raw MediaPipe model classification before mirror adjustment
    score: float  # Confidence score in [0.0, 1.0]
    landmarks: List[Tuple[float, float, float]]  # 21 (x, y, z) normalized coordinates
    flat_landmarks: np.ndarray  # 1D array of shape (63,) with dtype float32
    world_landmarks: Optional[List[Tuple[float, float, float]]] = None


class HandDetector:
    """Wrapper around MediaPipe HandLandmarker for real-time video processing."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Initialize the HandDetector.

        Args:
            model_path: Path to the hand_landmarker.task bundle.
            num_hands: Maximum number of hands to detect concurrently.
            min_hand_detection_confidence: Minimum confidence for hand detection.
            min_hand_presence_confidence: Minimum confidence for hand presence.
            min_tracking_confidence: Minimum confidence for landmark tracking.
        """
        self.model_path = Path(model_path)
        self.num_hands = num_hands
        self.min_detection_confidence = min_hand_detection_confidence
        self.min_presence_confidence = min_hand_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Ensure the model file is present before initializing MediaPipe
        self._ensure_model_exists()

        # Build MediaPipe HandLandmarkerOptions with RunningMode.VIDEO
        # VIDEO mode provides temporal tracking between successive frames.
        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        self._landmarker: Optional[vision.HandLandmarker] = (
            vision.HandLandmarker.create_from_options(options)
        )

        # In VIDEO mode, timestamps MUST be strictly monotonically increasing.
        self._last_timestamp_ms: int = -1

        # Caching drawing styles and connections for rendering
        self._connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
        self._landmarks_style = (
            vision.drawing_styles.get_default_hand_landmarks_style()
        )
        self._connections_style = (
            vision.drawing_styles.get_default_hand_connections_style()
        )

    def _ensure_model_exists(self) -> None:
        """Verify the model file exists; offer automated download if missing."""
        if self.model_path.is_file():
            return

        print(f"[HandDetector] Model file not found at: {self.model_path.resolve()}")
        print(f"[HandDetector] Downloading pretrained model from {DEFAULT_MODEL_URL}...")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Download model asset directly
            urllib.request.urlretrieve(DEFAULT_MODEL_URL, str(self.model_path))
            print(f"[HandDetector] Download complete: {self.model_path.resolve()} "
                  f"({self.model_path.stat().st_size / (1024 * 1024):.2f} MB)")
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to download model to {self.model_path}.\n"
                f"Please manually download it using:\n"
                f"  mkdir -p {self.model_path.parent}\n"
                f"  curl -L -o {self.model_path} {DEFAULT_MODEL_URL}\n"
                f"Original error: {e}"
            ) from e

    def detect(
        self, bgr_frame: np.ndarray, timestamp_ms: Optional[int] = None
    ) -> vision.HandLandmarkerResult:
        """Run hand landmark detection on an OpenCV BGR frame.

        Args:
            bgr_frame: Frame in BGR format as captured by OpenCV.
            timestamp_ms: Optional timestamp in milliseconds. If None, current
                monotonic time will be calculated and guaranteed to strictly increase.

        Returns:
            HandLandmarkerResult containing detected hand landmarks and handedness.
        """
        if self._landmarker is None:
            raise RuntimeError("HandDetector is closed or not initialized.")

        # OpenCV frames are in BGR, MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Maintain strictly monotonically increasing timestamp for VIDEO mode
        if timestamp_ms is None:
            curr_ms = int(time.time() * 1000)
            if curr_ms <= self._last_timestamp_ms:
                curr_ms = self._last_timestamp_ms + 1
            timestamp_ms = curr_ms
        elif timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1

        self._last_timestamp_ms = timestamp_ms

        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    def extract_landmarks(
        self,
        detection_result: vision.HandLandmarkerResult,
        is_mirrored: bool = True,
    ) -> List[HandLandmarkData]:
        """Extract structured normalized landmark features from detection results.

        Each detected hand produces 21 landmarks:
        - `x`: normalized to [0.0, 1.0] relative to image width
        - `y`: normalized to [0.0, 1.0] relative to image height
        - `z`: landmark depth relative to wrist (smaller means closer to camera)

        The extracted 1D `flat_landmarks` array has shape (63,) containing:
        [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20].
        This format is directly compatible with downstream machine learning
        classifiers (e.g., SVM, MLP, Random Forest).

        Handedness Diagnosis:
        MediaPipe outputs handedness from an observer's perspective assuming an
        unmirrored input. When a webcam frame is horizontally flipped (selfie/mirror
        mode), the 2D reflection inverts the hand's chirality (reflection of a physical
        right hand has the geometric properties of a left hand). When `is_mirrored` is
        True, this method corrects the handedness label to match the user's physical hand.

        Args:
            detection_result: Result from `detect()`.
            is_mirrored: Whether the frame fed to `detect()` was horizontally flipped.

        Returns:
            List of `HandLandmarkData` instances, one per detected hand.
        """
        extracted_hands: List[HandLandmarkData] = []

        if not detection_result.hand_landmarks:
            return extracted_hands

        num_detected = len(detection_result.hand_landmarks)

        for i in range(num_detected):
            raw_landmarks = detection_result.hand_landmarks[i]

            # Parse handedness metadata if provided
            raw_label = "Unknown"
            score = 0.0
            if detection_result.handedness and i < len(detection_result.handedness):
                categories = detection_result.handedness[i]
                if categories:
                    raw_label = categories[0].category_name
                    score = float(categories[0].score)

            # Correct handedness for mirrored input feeds
            if is_mirrored and raw_label in ("Left", "Right"):
                physical_handedness = "Right" if raw_label == "Left" else "Left"
            else:
                physical_handedness = raw_label

            # Extract 21 (x, y, z) tuples
            coords: List[Tuple[float, float, float]] = [
                (lm.x, lm.y, lm.z) for lm in raw_landmarks
            ]

            # Flatten to 63-element feature vector for classification models
            flat_vector = np.array(coords, dtype=np.float32).flatten()

            # Optional real-world 3D coordinates (in meters)
            world_coords: Optional[List[Tuple[float, float, float]]] = None
            if (
                detection_result.hand_world_landmarks
                and i < len(detection_result.hand_world_landmarks)
            ):
                world_landmarks = detection_result.hand_world_landmarks[i]
                world_coords = [(lm.x, lm.y, lm.z) for lm in world_landmarks]

            extracted_hands.append(
                HandLandmarkData(
                    handedness=physical_handedness,
                    raw_handedness=raw_label,
                    score=score,
                    landmarks=coords,
                    flat_landmarks=flat_vector,
                    world_landmarks=world_coords,
                )
            )

        return extracted_hands

    def draw_landmarks(
        self,
        bgr_frame: np.ndarray,
        detection_result: vision.HandLandmarkerResult,
        is_mirrored: bool = True,
        draw_labels: bool = True,
    ) -> np.ndarray:
        """Visualize detected 21 landmarks, connections, and handedness on frame.

        Args:
            bgr_frame: Image frame in BGR format to draw on (modified in place).
            detection_result: Detection result from `detect()`.
            is_mirrored: Whether the frame was horizontally flipped (for handedness label).
            draw_labels: Whether to render handedness and confidence score tags.

        Returns:
            The annotated frame (for method chaining).
        """
        if not detection_result.hand_landmarks:
            return bgr_frame

        h, w, _ = bgr_frame.shape

        for i, landmarks in enumerate(detection_result.hand_landmarks):
            # Render landmarks and connections using MediaPipe drawing utilities
            vision.drawing_utils.draw_landmarks(
                image=bgr_frame,
                landmark_list=landmarks,
                connections=self._connections,
                landmark_drawing_spec=self._landmarks_style,
                connection_drawing_spec=self._connections_style,
            )

            # Render hand label (e.g. "Right: 96%") near wrist (landmark 0)
            if draw_labels and landmarks:
                wrist = landmarks[0]
                wrist_px = (int(wrist.x * w), int(wrist.y * h))

                label = "Hand"
                if detection_result.handedness and i < len(detection_result.handedness):
                    cat = detection_result.handedness[i][0]
                    raw_name = cat.category_name
                    # Mirror correction for display
                    if is_mirrored and raw_name in ("Left", "Right"):
                        display_name = "Right" if raw_name == "Left" else "Left"
                    else:
                        display_name = raw_name
                    label = f"{display_name} ({cat.score:.0%})"

                # Background banner for text readability
                label_pos = (max(10, wrist_px[0] - 40), max(25, wrist_px[1] - 15))
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                )
                cv2.rectangle(
                    bgr_frame,
                    (label_pos[0] - 2, label_pos[1] - th - 4),
                    (label_pos[0] + tw + 2, label_pos[1] + baseline),
                    (20, 20, 20),
                    cv2.FILLED,
                )
                cv2.putText(
                    bgr_frame,
                    label,
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 200),
                    1,
                    cv2.LINE_AA,
                )

        return bgr_frame

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> HandDetector:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def render_hud(
    frame: np.ndarray,
    fps: float,
    inference_ms: float,
    num_hands: int,
    hand_details: List[HandLandmarkData],
) -> None:
    """Render a clean information overlay (HUD) using optimized ROI blending.

    Avoids copying the full frame buffer by applying alpha blending only to the
    HUD region of interest (ROI), which significantly reduces CPU memory overhead.
    """
    hud_h = 75
    hud_w = 320
    x, y = 10, 10

    # Ensure ROI fits within frame bounds
    h_frame, w_frame, _ = frame.shape
    if y + hud_h <= h_frame and x + hud_w <= w_frame:
        hud_roi = frame[y : y + hud_h, x : x + hud_w]
        # Fast ROI-only alpha blend (avoids full-frame copy)
        dark_rect = np.full_like(hud_roi, 15)
        frame[y : y + hud_h, x : x + hud_w] = cv2.addWeighted(
            dark_rect, 0.65, hud_roi, 0.35, 0
        )
        # Subtle border
        cv2.rectangle(frame, (x, y), (x + hud_w, y + hud_h), (70, 70, 70), 1)

    # Line 1: FPS and Inference Latency (reported separately for CPU monitoring)
    fps_color = (0, 255, 0) if fps >= 25 else ((0, 255, 255) if fps >= 15 else (0, 0, 255))
    metrics_text = f"FPS: {fps:4.1f} | Inf: {inference_ms:4.1f}ms (CPU)"
    cv2.putText(
        frame,
        metrics_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        fps_color,
        2,
        cv2.LINE_AA,
    )

    # Line 2: Hands detected and physical handedness breakdown
    info_text = f"Hands: {num_hands}"
    if hand_details:
        labels = [f"{h.handedness[:1]}:{h.score:.2f}" for h in hand_details]
        info_text += f" ({', '.join(labels)})"

    cv2.putText(
        frame,
        info_text,
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    # Instructions in bottom left
    cv2.putText(
        frame,
        "Press 'q'/Esc to quit | 'p' to log features",
        (15, h_frame - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )


def open_webcam(
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
) -> Optional[cv2.VideoCapture]:
    """Open and configure the webcam with error diagnostics.

    Defaults to 1280x720 (native 16:9) capture mode to align with native camera
    sensors and avoid driver-level aspect ratio scaling overhead.

    Args:
        camera_index: Device index for cv2.VideoCapture.
        width: Requested capture width (default: 1280).
        height: Requested capture height (default: 720).

    Returns:
        Configured cv2.VideoCapture object or None if failed.
    """
    print(f"[Webcam] Attempting to open camera index {camera_index} at {width}x{height}...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"\n[Error] Unable to access camera at index {camera_index}!", file=sys.stderr)
        print("Diagnostic checklist:", file=sys.stderr)
        print(" 1. Check if another application (Zoom, FaceTime, Browser) is using the camera.", file=sys.stderr)
        print(" 2. On macOS, ensure your Terminal / IDE has Camera permission in:", file=sys.stderr)
        print("    System Settings -> Privacy & Security -> Camera.", file=sys.stderr)
        print(" 3. Try passing a different camera index: --camera 1 (or 2).", file=sys.stderr)
        return None

    # Request resolution from webcam hardware
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Read a test frame to verify actual image retrieval
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("[Error] Camera opened but failed to capture the first frame.", file=sys.stderr)
        cap.release()
        return None

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Webcam] Successfully opened camera {camera_index} ({actual_w}x{actual_h}).")
    return cap


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-Time Hand Landmark Detection using MediaPipe Tasks Vision"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help=f"Path to hand_landmarker.task model file (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Webcam capture width (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Webcam capture height (default: 720)",
    )
    parser.add_argument(
        "--num-hands",
        type=int,
        default=2,
        help="Maximum number of hands to detect concurrently (default: 2)",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for hand detection in [0.0, 1.0] (default: 0.5)",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for landmark tracking in [0.0, 1.0] (default: 0.5)",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal flip (default displays mirror/selfie mode)",
    )
    parser.add_argument(
        "--print-features",
        action="store_true",
        help="Print extracted 63-D landmark feature vector summary to stdout",
    )
    parser.add_argument(
        "--test-synthetic",
        action="store_true",
        help="Internal test mode: run detection on synthetic frames without webcam",
    )
    return parser.parse_args()


def run_synthetic_test(detector: HandDetector) -> None:
    """Run an automated self-test on simulated frames for environments without webcam."""
    print("\n--- Running Synthetic Test (No Webcam) ---")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a mock circular patch
    cv2.circle(dummy_frame, (320, 240), 50, (200, 200, 200), -1)

    t0 = time.perf_counter()
    result = detector.detect(dummy_frame)
    inf_ms = (time.perf_counter() - t0) * 1000

    extracted = detector.extract_landmarks(result, is_mirrored=True)
    detector.draw_landmarks(dummy_frame, result, is_mirrored=True)
    render_hud(dummy_frame, fps=45.0, inference_ms=inf_ms, num_hands=len(extracted), hand_details=extracted)

    print(f"Test frame (640x480) processed successfully.")
    print(f"Inference latency on synthetic frame: {inf_ms:.2f} ms")
    print(f"Hands detected on blank test frame: {len(extracted)}")
    print("Synthetic test PASSED.\n")


def main() -> None:
    """Main entry point for hand gesture landmark detector."""
    args = parse_arguments()

    print("=" * 65)
    print("Hand Gesture Recognition - Landmark Detector (MediaPipe Tasks)")
    print("=" * 65)

    # Initialize the HandDetector with selected parameters
    try:
        detector = HandDetector(
            model_path=args.model,
            num_hands=args.num_hands,
            min_hand_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
    except Exception as e:
        print(f"[Fatal] Failed to initialize HandDetector: {e}", file=sys.stderr)
        sys.exit(1)

    # Optional synthetic test execution (e.g. for CI or headless smoke tests)
    if args.test_synthetic:
        with detector:
            run_synthetic_test(detector)
        return

    # Open webcam with requested resolution (default: 640x480)
    cap = open_webcam(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
    )
    if cap is None:
        detector.close()
        sys.exit(1)

    window_name = "Hand Gesture Recognition - Landmark Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Performance monitoring variables
    prev_time = time.perf_counter()
    fps_smooth = 0.0
    inf_ms_smooth = 0.0
    alpha = 0.1  # Exponential moving average smoothing factor

    is_mirrored = not args.no_mirror

    print(f"\nConfiguration:")
    print(f" - Resolution: {args.width}x{args.height}")
    print(f" - Mode: {'Mirrored (Selfie)' if is_mirrored else 'Unmirrored (Standard)'}")
    print(f" - Max hands: {args.num_hands}")
    print(f"\nStarting real-time hand detection loop.")
    print("Controls:")
    print(" - Press 'q' or 'ESC' to exit")
    print(" - Press 'p' to toggle landmark feature printing\n")

    print_features_active = args.print_features

    try:
        with detector:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[Warning] Failed to receive frame from webcam. Stream ended.")
                    break

                # Mirror frame horizontally for intuitive interaction (selfie mode)
                if is_mirrored:
                    frame = cv2.flip(frame, 1)

                # 1. Detect hand landmarks (timed separately for pure CPU inference latency)
                t_inf_start = time.perf_counter()
                detection_result = detector.detect(frame)
                t_inf_end = time.perf_counter()
                curr_inf_ms = (t_inf_end - t_inf_start) * 1000

                # 2. Extract structured landmarks (with mirror-aware handedness correction)
                extracted_hands = detector.extract_landmarks(
                    detection_result, is_mirrored=is_mirrored
                )

                # 3. Visualize landmarks and connections on frame
                detector.draw_landmarks(
                    frame, detection_result, is_mirrored=is_mirrored, draw_labels=True
                )

                # Calculate smoothed pipeline FPS and inference latency
                curr_time = time.perf_counter()
                frame_delta = curr_time - prev_time
                prev_time = curr_time
                if frame_delta > 0:
                    current_fps = 1.0 / frame_delta
                    fps_smooth = (
                        current_fps
                        if fps_smooth == 0.0
                        else (alpha * current_fps + (1.0 - alpha) * fps_smooth)
                    )

                inf_ms_smooth = (
                    curr_inf_ms
                    if inf_ms_smooth == 0.0
                    else (alpha * curr_inf_ms + (1.0 - alpha) * inf_ms_smooth)
                )

                # 4. Render optimized HUD with separate FPS and inference latency
                render_hud(
                    frame=frame,
                    fps=fps_smooth,
                    inference_ms=inf_ms_smooth,
                    num_hands=len(extracted_hands),
                    hand_details=extracted_hands,
                )

                # Optional debug logging for classifier features
                if print_features_active and extracted_hands:
                    hand0 = extracted_hands[0]
                    # Display first 6 coordinates (wrist & thumb CMC) as demonstration
                    preview = np.round(hand0.flat_landmarks[:6], 3)
                    print(
                        f"\r[{hand0.handedness} hand | Score: {hand0.score:.2f} | Inf: {curr_inf_ms:.1f}ms] "
                        f"Features (shape {hand0.flat_landmarks.shape}): {preview}...",
                        end="",
                        flush=True,
                    )

                # Display frame
                cv2.imshow(window_name, frame)

                # Handle keyboard inputs
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 27 is ESC
                    print("\n[Exit] User requested termination.")
                    break
                elif key in (ord("p"), ord("P")):
                    print_features_active = not print_features_active
                    status = "ON" if print_features_active else "OFF"
                    print(f"\n[Feature Logging] Feature vector logging: {status}")

    except KeyboardInterrupt:
        print("\n[Exit] Interrupted by user (Ctrl+C).")
    finally:
        # Proper resource cleanup
        print("[Cleanup] Releasing camera and closing windows...")
        cap.release()
        cv2.destroyAllWindows()
        print("[Cleanup] Done.")


if __name__ == "__main__":
    main()

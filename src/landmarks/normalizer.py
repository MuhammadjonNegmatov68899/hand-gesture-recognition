"""Landmark coordinate normalization for hand gesture recognition.

This module implements translation and scale normalization for 21 3D hand
landmarks predicted by MediaPipe Tasks Hand Landmarker.

Methodology for Scientific Description:
--------------------------------------
1. Translation Normalization (Wrist-Centric):
   Given 21 landmarks P = [p_0, p_1, ..., p_20]^T where p_i = (x_i, y_i, z_i),
   we achieve translation invariance by subtracting the wrist position (p_0):
       p'_i = p_i - p_0  for all i in {0, ..., 20}
   After translation, p'_0 = (0, 0, 0).

2. Scale Normalization (Anatomical Reference Distance):
   To achieve scale invariance across varying distances from camera and hand
   sizes, we calculate the Euclidean distance between the wrist (p_0) and
   the middle fingertip (landmark 12, p_12):
       d_ref = ||p'_12||_2 = sqrt(x'^2_12 + y'^2_12 + z'^2_12)
   Each centered coordinate is normalized by the reference distance:
       p''_i = p'_i / (d_ref + eps)
   If d_ref < eps (degenerate or zero-length hand), coordinates are retained
   without division to avoid division-by-zero, NaN, or infinite values.

3. Feature Vector:
   The resulting (21, 3) matrix is flattened into a 63-dimensional feature
   vector with float32 precision:
       [x''_0, y''_0, z''_0, x''_1, y''_1, z''_1, ..., x''_20, y''_20, z''_20]
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple, Union
import numpy as np


class LandmarkNormalizer:
    """Normalizes hand landmarks to achieve translation and scale invariance."""

    def __init__(
        self,
        reference_landmark_index: int = 12,
        eps: float = 1e-7,
    ) -> None:
        """Initialize LandmarkNormalizer.

        Args:
            reference_landmark_index: Landmark index used for scale reference.
                Default is 12 (MIDDLE_FINGER_TIP).
            eps: Small epsilon constant to prevent division by zero.
        """
        if not (0 <= reference_landmark_index < 21):
            raise ValueError(
                f"reference_landmark_index must be between 0 and 20, got {reference_landmark_index}"
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.reference_landmark_index = reference_landmark_index
        self.eps = float(eps)

    def _validate_and_convert(self, landmarks: Any) -> Tuple[np.ndarray, bool]:
        """Validate input and convert to a numpy array of shape (N, 21, 3).

        Args:
            landmarks: Input landmarks. Can be:
                - HandLandmarkData instance
                - List of 21 (x, y, z) tuples
                - 1D array of shape (63,)
                - 2D array of shape (21, 3) (single hand)
                - 2D array of shape (N, 63) (batch)
                - 3D array of shape (N, 21, 3) (batch)

        Returns:
            Tuple of:
                - array of shape (N, 21, 3) with float32 dtype
                - is_batch: boolean indicating if input was a batch

        Raises:
            TypeError: If input is None or cannot be parsed as numeric array.
            ValueError: If input shape is invalid or contains NaN/Inf values.
        """
        if landmarks is None:
            raise TypeError("Landmarks input cannot be None.")

        # Check for HandLandmarkData duck-typing (single hand or list of hands)
        if hasattr(landmarks, "flat_landmarks"):
            landmarks = landmarks.flat_landmarks
        elif hasattr(landmarks, "landmarks") and isinstance(landmarks.landmarks, (list, tuple)):
            landmarks = landmarks.landmarks
        elif isinstance(landmarks, (list, tuple)) and len(landmarks) > 0:
            if hasattr(landmarks[0], "flat_landmarks"):
                landmarks = [h.flat_landmarks for h in landmarks]
            elif hasattr(landmarks[0], "landmarks") and isinstance(landmarks[0].landmarks, (list, tuple)):
                landmarks = [h.landmarks for h in landmarks]

        try:
            arr = np.asarray(landmarks, dtype=np.float32)
        except (ValueError, TypeError) as err:
            raise TypeError(f"Could not convert landmarks to float32 numpy array: {err}") from err

        if arr.size == 0:
            raise ValueError("Landmarks array cannot be empty.")

        # Check for NaN and Inf values
        if not np.isfinite(arr).all():
            raise ValueError("Landmarks contain NaN, Inf, or invalid numeric values.")

        # Determine shape and batch status
        if arr.ndim == 1:
            if arr.shape[0] != 63:
                raise ValueError(
                    f"1D landmarks array must have length 63, got shape {arr.shape}."
                )
            arr = arr.reshape(1, 21, 3)
            is_batch = False
        elif arr.ndim == 2:
            if arr.shape == (21, 3):
                arr = arr.reshape(1, 21, 3)
                is_batch = False
            elif arr.shape[1] == 63:
                arr = arr.reshape(-1, 21, 3)
                is_batch = True
            else:
                raise ValueError(
                    f"Invalid landmarks array shape: {arr.shape}. Expected (21, 3) for single hand or "
                    f"(N, 63) for batch."
                )
        elif arr.ndim == 3:
            if arr.shape[1:] != (21, 3):
                raise ValueError(
                    f"Invalid landmarks array shape: {arr.shape}. Expected (N, 21, 3)."
                )
            is_batch = True
        else:
            raise ValueError(
                f"Invalid landmarks array dimensions: ndim={arr.ndim}. "
                f"Expected 1D (63,), 2D (21, 3) or (N, 63), or 3D (N, 21, 3)."
            )

        return arr, is_batch

    def normalize(
        self,
        landmarks: Any,
    ) -> np.ndarray:
        """Normalize landmarks for single hand or batch of hands.

        Args:
            landmarks: 21 landmarks for a single hand or batch of hands.
                Accepts:
                - 1D array of shape (63,) -> returns 1D array (63,)
                - 2D array of shape (21, 3) -> returns 1D array (63,)
                - List of 21 (x, y, z) tuples -> returns 1D array (63,)
                - HandLandmarkData -> returns 1D array (63,)
                - 2D array of shape (N, 63) -> returns 2D array (N, 63)
                - 3D array of shape (N, 21, 3) -> returns 2D array (N, 63)

        Returns:
            Normalized 63-dimensional float32 feature vector (shape (63,)
            for single hand or (N, 63) for batch).
        """
        arr, is_batch = self._validate_and_convert(landmarks)

        # 1. Wrist-relative translation normalization: p'_i = p_i - p_0
        wrist = arr[:, 0:1, :]  # shape: (N, 1, 3)
        centered = arr - wrist  # shape: (N, 21, 3)

        # 2. Scale normalization: d_ref = ||p'_ref||_2
        ref_landmark = centered[:, self.reference_landmark_index, :]  # shape: (N, 3)
        ref_distance = np.linalg.norm(ref_landmark, axis=-1, keepdims=True)  # shape: (N, 1)

        # Avoid division by zero: if ref_distance < eps, scale by 1.0 (no scaling)
        # Otherwise divide by (ref_distance + eps)
        safe_scale = np.where(
            ref_distance < self.eps,
            1.0,
            ref_distance + self.eps,
        )  # shape: (N, 1)

        safe_scale = safe_scale[:, :, np.newaxis]  # shape: (N, 1, 1)
        normalized = centered / safe_scale  # shape: (N, 21, 3)

        # Ensure float32 dtype
        normalized = normalized.astype(np.float32)

        # Reshape to feature vector: (N, 63)
        features = normalized.reshape(arr.shape[0], 63)

        if not is_batch:
            return features[0]
        return features

    def normalize_hand(self, landmarks: Any) -> np.ndarray:
        """Normalize a single hand's landmarks to a 1D (63,) float32 vector."""
        res = self.normalize(landmarks)
        if res.ndim != 1:
            if res.shape[0] == 1:
                return res[0]
            raise ValueError(
                f"Expected single hand output (63,), got batch shape {res.shape}."
            )
        return res

    def normalize_batch(self, batch_landmarks: Any) -> np.ndarray:
        """Normalize a batch of hands to a 2D (N, 63) float32 array."""
        arr, _ = self._validate_and_convert(batch_landmarks)
        # Force batch mode even if N=1
        features = self.normalize(arr)
        if features.ndim == 1:
            return features.reshape(1, 63)
        return features

"""Unit tests for LandmarkNormalizer."""

import numpy as np
import pytest

from src.landmarks.hand_detector import HandLandmarkData
from src.landmarks.normalizer import LandmarkNormalizer


def test_normalizer_initialization():
    normalizer = LandmarkNormalizer()
    assert normalizer.reference_landmark_index == 12
    assert normalizer.eps == 1e-7

    with pytest.raises(ValueError):
        LandmarkNormalizer(reference_landmark_index=-1)
    with pytest.raises(ValueError):
        LandmarkNormalizer(reference_landmark_index=21)
    with pytest.raises(ValueError):
        LandmarkNormalizer(eps=-1e-5)
    with pytest.raises(ValueError):
        LandmarkNormalizer(eps=0)


def test_normalize_single_hand_shape_21_3():
    normalizer = LandmarkNormalizer()
    # Create arbitrary 21 landmarks
    raw_landmarks = np.arange(63, dtype=np.float32).reshape(21, 3)

    output = normalizer.normalize(raw_landmarks)
    assert isinstance(output, np.ndarray)
    assert output.shape == (63,)
    assert output.dtype == np.float32


def test_normalize_single_hand_shape_63():
    normalizer = LandmarkNormalizer()
    raw_flat = np.linspace(0.0, 1.0, 63, dtype=np.float32)

    output = normalizer.normalize(raw_flat)
    assert output.shape == (63,)
    assert output.dtype == np.float32


def test_normalize_hand_data_dataclass():
    normalizer = LandmarkNormalizer()
    raw_coords = [(float(i), float(i + 1), float(i + 2)) for i in range(21)]
    flat_coords = np.array(raw_coords, dtype=np.float32).flatten()
    data = HandLandmarkData(
        handedness="Right",
        raw_handedness="Left",
        score=0.95,
        landmarks=raw_coords,
        flat_landmarks=flat_coords,
    )

    # Single hand
    output = normalizer.normalize(data)
    assert output.shape == (63,)
    assert output.dtype == np.float32

    # Multiple hands in list (batch)
    raw_coords2 = [(float(i * 2), float(i * 2 + 1), float(i * 2 + 2)) for i in range(21)]
    flat_coords2 = np.array(raw_coords2, dtype=np.float32).flatten()
    data2 = HandLandmarkData(
        handedness="Left",
        raw_handedness="Right",
        score=0.90,
        landmarks=raw_coords2,
        flat_landmarks=flat_coords2,
    )

    batch_output = normalizer.normalize([data, data2])
    assert isinstance(batch_output, np.ndarray)
    assert batch_output.shape == (2, 63)
    assert batch_output.dtype == np.float32

    # normalize_batch method with list of HandLandmarkData
    batch_method_out = normalizer.normalize_batch([data, data2])
    assert batch_method_out.shape == (2, 63)
    assert batch_method_out.dtype == np.float32


def test_normalize_batch_shapes():
    normalizer = LandmarkNormalizer()
    N = 10

    # Batch of shape (N, 21, 3)
    batch_3d = np.random.RandomState(42).uniform(-1.0, 1.0, size=(N, 21, 3)).astype(np.float32)
    out_3d = normalizer.normalize(batch_3d)
    assert out_3d.shape == (N, 63)
    assert out_3d.dtype == np.float32

    # Batch of shape (N, 63)
    batch_2d = np.random.RandomState(42).uniform(-1.0, 1.0, size=(N, 63)).astype(np.float32)
    out_2d = normalizer.normalize(batch_2d)
    assert out_2d.shape == (N, 63)
    assert out_2d.dtype == np.float32

    # normalize_batch method
    out_batch_method = normalizer.normalize_batch(batch_3d)
    assert out_batch_method.shape == (N, 63)


def test_wrist_relative_translation():
    normalizer = LandmarkNormalizer()
    # Wrist at (10.0, 20.0, 30.0)
    raw_landmarks = np.zeros((21, 3), dtype=np.float32)
    raw_landmarks[0] = [10.0, 20.0, 30.0]
    raw_landmarks[12] = [10.0, 20.0, 32.0]  # distance = 2.0 along z

    normalized = normalizer.normalize(raw_landmarks).reshape(21, 3)

    # Landmark 0 (wrist) should be exactly (0.0, 0.0, 0.0)
    np.testing.assert_allclose(normalized[0], [0.0, 0.0, 0.0], atol=1e-6)

    # Relative position of landmark 12 should be along z axis
    assert normalized[12, 0] == 0.0
    assert normalized[12, 1] == 0.0
    assert normalized[12, 2] > 0.0


def test_scale_normalization():
    normalizer = LandmarkNormalizer()
    raw_landmarks = np.zeros((21, 3), dtype=np.float32)
    raw_landmarks[0] = [0.0, 0.0, 0.0]
    # Middle fingertip (landmark 12) at distance 5.0 from wrist
    raw_landmarks[12] = [3.0, 4.0, 0.0]  # norm = 5.0
    raw_landmarks[5] = [1.0, 2.0, 0.0]

    normalized = normalizer.normalize(raw_landmarks).reshape(21, 3)

    # Norm of landmark 12 should be approximately 1.0 (5.0 / (5.0 + 1e-7) ≈ 1.0)
    ref_norm = np.linalg.norm(normalized[12])
    np.testing.assert_allclose(ref_norm, 1.0, atol=1e-5)

    # Translation invariance: shifting raw landmarks by an offset should yield same normalized features
    offset = np.array([100.0, -50.0, 25.0], dtype=np.float32)
    shifted_landmarks = raw_landmarks + offset
    normalized_shifted = normalizer.normalize(shifted_landmarks)

    np.testing.assert_allclose(normalized.flatten(), normalized_shifted, atol=1e-6)


def test_zero_distance_handling():
    normalizer = LandmarkNormalizer()
    # Degenerate hand: all landmarks at the exact same location
    raw_landmarks = np.ones((21, 3), dtype=np.float32) * 5.0

    normalized = normalizer.normalize(raw_landmarks)

    # Should not raise exception or produce NaNs/Infs
    assert not np.isnan(normalized).any()
    assert not np.isinf(normalized).any()
    # Centered coordinates should all be 0.0
    np.testing.assert_allclose(normalized, 0.0, atol=1e-6)


def test_nan_and_infinite_input_handling():
    normalizer = LandmarkNormalizer()

    # Input containing NaN
    raw_nan = np.zeros((21, 3), dtype=np.float32)
    raw_nan[5, 1] = np.nan
    with pytest.raises(ValueError, match="NaN, Inf"):
        normalizer.normalize(raw_nan)

    # Input containing positive infinity
    raw_inf = np.zeros((21, 3), dtype=np.float32)
    raw_inf[10, 0] = np.inf
    with pytest.raises(ValueError, match="NaN, Inf"):
        normalizer.normalize(raw_inf)

    # Input containing negative infinity
    raw_neg_inf = np.zeros((21, 3), dtype=np.float32)
    raw_neg_inf[12, 2] = -np.inf
    with pytest.raises(ValueError, match="NaN, Inf"):
        normalizer.normalize(raw_neg_inf)


def test_invalid_input_types_and_shapes():
    normalizer = LandmarkNormalizer()

    # None
    with pytest.raises(TypeError):
        normalizer.normalize(None)

    # Empty array
    with pytest.raises(ValueError, match="empty"):
        normalizer.normalize(np.array([]))

    # Wrong shape (e.g. 20 landmarks instead of 21)
    with pytest.raises(ValueError, match="Invalid landmarks"):
        normalizer.normalize(np.zeros((20, 3)))

    # Wrong dimension (e.g. 4 coordinates)
    with pytest.raises(ValueError, match="Invalid landmarks"):
        normalizer.normalize(np.zeros((21, 4)))

    # 1D wrong length (e.g. 60 instead of 63)
    with pytest.raises(ValueError, match="length 63"):
        normalizer.normalize(np.zeros(60))

    # 4D array
    with pytest.raises(ValueError, match="Invalid landmarks array dimensions"):
        normalizer.normalize(np.zeros((2, 3, 21, 3)))


def test_deterministic_normalization():
    normalizer = LandmarkNormalizer()
    rng = np.random.RandomState(12345)
    sample = rng.uniform(-10.0, 10.0, size=(21, 3)).astype(np.float32)

    res1 = normalizer.normalize(sample)
    res2 = normalizer.normalize(sample)

    np.testing.assert_array_equal(res1, res2)

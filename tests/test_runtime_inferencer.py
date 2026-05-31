"""Tests for the ONNX runtime inferencer (pure helpers, no onnxruntime needed)."""

import pytest

from alchemydetect.core.runtime_inferencer import compute_resize_scale


def test_resize_scales_up_shortest_edge():
    # Shortest edge 400 -> 800; longest edge 600*2=1200 within max.
    assert compute_resize_scale(400, 600, min_size=800, max_size=1333) == pytest.approx(2.0)


def test_resize_clamped_by_max_size():
    # Scaling the shortest edge would push the longest edge past max_size,
    # so the scale is clamped by max_size instead.
    scale = compute_resize_scale(1000, 2000, min_size=800, max_size=1333)
    assert scale == pytest.approx(1333 / 2000)
    assert max(1000, 2000) * scale <= 1333 + 1e-6


def test_resize_square_image():
    assert compute_resize_scale(500, 500, min_size=800, max_size=1333) == pytest.approx(1.6)


def test_resize_downscales_large_image():
    # A large image whose shortest edge already exceeds min_size is scaled down.
    scale = compute_resize_scale(2000, 3000, min_size=800, max_size=1333)
    assert scale < 1.0
    assert max(2000, 3000) * scale <= 1333 + 1e-6

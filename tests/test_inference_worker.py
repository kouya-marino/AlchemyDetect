"""Tests for inference worker module."""

import os

import cv2

from alchemydetect.workers.inference_worker import InferenceWorker


def test_collect_image_paths_single_file(temp_dir, sample_image_bgr):
    img_path = os.path.join(temp_dir, "test.jpg")
    cv2.imwrite(img_path, sample_image_bgr)
    paths = InferenceWorker.collect_image_paths(img_path)
    assert len(paths) == 1


def test_collect_image_paths_directory(temp_dir, sample_image_bgr):
    for name in ["a.jpg", "b.png", "c.txt", "d.bmp"]:
        path = os.path.join(temp_dir, name)
        if name.endswith(".txt"):
            with open(path, "w") as f:
                f.write("not an image")
        else:
            cv2.imwrite(path, sample_image_bgr)
    paths = InferenceWorker.collect_image_paths(temp_dir)
    assert len(paths) == 3  # jpg, png, bmp — not txt


def test_collect_image_paths_empty_dir(temp_dir):
    paths = InferenceWorker.collect_image_paths(temp_dir)
    assert len(paths) == 0


def test_collect_image_paths_nonexistent():
    paths = InferenceWorker.collect_image_paths("/nonexistent/path")
    assert len(paths) == 0


def test_image_extensions():
    expected = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    assert InferenceWorker.IMAGE_EXTENSIONS == expected

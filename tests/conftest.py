"""Shared test fixtures."""

import json
import os
import tempfile

import numpy as np
import pytest


@pytest.fixture
def sample_image_bgr():
    """Generate a random 100x100 BGR uint8 image."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def temp_dir():
    """Create a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def coco_dataset(temp_dir, sample_image_bgr):
    """Create a minimal COCO dataset in a temp directory."""
    import cv2

    images_dir = os.path.join(temp_dir, "images")
    os.makedirs(images_dir)

    # Save a sample image
    img_path = os.path.join(images_dir, "test_001.jpg")
    cv2.imwrite(img_path, sample_image_bgr)

    # Create COCO JSON
    coco_data = {
        "images": [{"id": 1, "file_name": "test_001.jpg", "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 30, 30],
                "area": 900,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "object", "supercategory": "thing"}],
    }

    json_path = os.path.join(temp_dir, "annotations.json")
    with open(json_path, "w") as f:
        json.dump(coco_data, f)

    return images_dir, json_path

"""Tests for dataset utilities module."""

import json
import os

from alchemydetect.core.dataset_utils import get_num_classes, validate_coco_json


def test_validate_coco_json_valid(coco_dataset):
    images_dir, json_path = coco_dataset
    is_valid, msg = validate_coco_json(json_path, images_dir)
    assert is_valid
    assert "1 images" in msg
    assert "1 categories" in msg


def test_validate_coco_json_missing_file(temp_dir):
    is_valid, msg = validate_coco_json("/nonexistent/path.json", temp_dir)
    assert not is_valid
    assert "not found" in msg


def test_validate_coco_json_missing_images_dir(coco_dataset):
    _, json_path = coco_dataset
    is_valid, msg = validate_coco_json(json_path, "/nonexistent/dir")
    assert not is_valid
    assert "not found" in msg


def test_validate_coco_json_invalid_json(temp_dir):
    bad_json = os.path.join(temp_dir, "bad.json")
    with open(bad_json, "w") as f:
        f.write("not valid json")
    is_valid, msg = validate_coco_json(bad_json, temp_dir)
    assert not is_valid
    assert "Invalid JSON" in msg


def test_validate_coco_json_missing_keys(temp_dir):
    incomplete_json = os.path.join(temp_dir, "incomplete.json")
    with open(incomplete_json, "w") as f:
        json.dump({"images": []}, f)
    is_valid, msg = validate_coco_json(incomplete_json, temp_dir)
    assert not is_valid
    assert "Missing required key" in msg


def test_validate_coco_json_empty_categories(temp_dir):
    empty_cats = os.path.join(temp_dir, "empty_cats.json")
    with open(empty_cats, "w") as f:
        json.dump({"images": [{"id": 1}], "annotations": [], "categories": []}, f)
    is_valid, msg = validate_coco_json(empty_cats, temp_dir)
    assert not is_valid
    assert "No categories" in msg


def test_get_num_classes(coco_dataset):
    _, json_path = coco_dataset
    assert get_num_classes(json_path) == 1

"""COCO dataset registration helpers for Detectron2."""

import json
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances


def register_coco_dataset(name, json_path, image_root):
    """Register a COCO-format dataset, skipping if already registered."""
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
        MetadataCatalog.remove(name)
    register_coco_instances(name, {}, json_path, image_root)


def validate_coco_json(json_path, image_root):
    """Validate a COCO JSON file. Returns (is_valid, error_message)."""
    json_path = Path(json_path)
    image_root = Path(image_root)

    if not json_path.exists():
        return False, f"Annotation file not found: {json_path}"

    if not image_root.exists():
        return False, f"Image directory not found: {image_root}"

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"

    for key in ("images", "annotations", "categories"):
        if key not in data:
            return False, f"Missing required key '{key}' in COCO JSON"

    if len(data["categories"]) == 0:
        return False, "No categories found in COCO JSON"

    if len(data["images"]) == 0:
        return False, "No images found in COCO JSON"

    # Check that at least some image files exist
    found = 0
    for img_info in data["images"][:10]:
        img_file = image_root / img_info["file_name"]
        if img_file.exists():
            found += 1

    if found == 0:
        return False, "None of the first 10 image files were found in the image directory"

    return True, f"Valid COCO dataset: {len(data['images'])} images, {len(data['categories'])} categories"


def get_num_classes(json_path):
    """Return the number of categories in a COCO JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return len(data["categories"])

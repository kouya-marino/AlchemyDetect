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

    # Every category needs an id and a name.
    for cat in data["categories"]:
        if "id" not in cat or "name" not in cat:
            return False, "Each category must have an 'id' and a 'name'"

    # Annotations must reference defined categories and carry a 4-number bbox, or
    # training fails later with a confusing error. Scan a bounded sample so this
    # pre-flight check stays snappy on very large datasets (a malformed export
    # shows up in the first handful of annotations).
    category_ids = {cat["id"] for cat in data["categories"]}
    for ann in data["annotations"][:2000]:
        if ann.get("category_id") not in category_ids:
            return False, f"Annotation {ann.get('id')} references unknown category_id {ann.get('category_id')}"
        bbox = ann.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False, f"Annotation {ann.get('id')} has an invalid bbox (expected 4 numbers)"

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


def get_class_names(json_path):
    """Return category names ordered to match Detectron2's contiguous class ids.

    Detectron2 (via the COCO API) maps category ids to contiguous ids 0..N-1 in
    ascending id order, so the names must be sorted by category id to line up
    with predicted class indices. Reading them in raw JSON order would mislabel
    predictions whenever categories are not already listed by ascending id.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    categories = sorted(data.get("categories", []), key=lambda c: c["id"])
    return [c["name"] for c in categories]


def get_dataset_summary(json_path):
    """Return a summary dict with image count, class count, class names, and annotation count."""
    with open(json_path, "r") as f:
        data = json.load(f)
    categories = sorted(data.get("categories", []), key=lambda c: c["id"])
    return {
        "num_images": len(data.get("images", [])),
        "num_annotations": len(data.get("annotations", [])),
        "num_classes": len(categories),
        "class_names": [c["name"] for c in categories],
    }

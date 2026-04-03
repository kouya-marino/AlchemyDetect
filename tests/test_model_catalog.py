"""Tests for model catalog module."""

from alchemydetect.core.model_catalog import (
    get_config_path,
    get_detection_models,
    get_model_names,
    get_segmentation_models,
    get_task,
)


def test_get_model_names():
    names = get_model_names()
    assert len(names) == 6
    assert "Faster R-CNN (R50-FPN)" in names
    assert "Mask R-CNN (R50-FPN)" in names


def test_get_detection_models():
    models = get_detection_models()
    assert len(models) == 4
    for name in models:
        assert get_task(name) == "detection"


def test_get_segmentation_models():
    models = get_segmentation_models()
    assert len(models) == 2
    for name in models:
        assert get_task(name) == "instance_segmentation"


def test_get_config_path():
    path = get_config_path("Faster R-CNN (R50-FPN)")
    assert "faster_rcnn" in path
    assert path.endswith(".yaml")


def test_get_task():
    assert get_task("Faster R-CNN (R50-FPN)") == "detection"
    assert get_task("Mask R-CNN (R50-FPN)") == "instance_segmentation"
    assert get_task("RetinaNet (R50-FPN)") == "detection"

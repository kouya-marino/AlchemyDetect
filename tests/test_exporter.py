"""Tests for the model exporter (pure helpers, no torch/detectron2 required)."""

import json
import os

import pytest

from alchemydetect.core.exporter import (
    build_export_metadata,
    copy_sidecar_files,
    detect_task_from_config,
    is_onnx_available,
    is_onnxruntime_available,
    is_tensorrt_available,
    read_class_names,
    resolve_model_dir,
)


def _make_model_dir(temp_dir, with_config=True, with_class_names=True):
    """Create a fake exported/trained model directory under temp_dir."""
    model_dir = os.path.join(temp_dir, "model")
    os.makedirs(model_dir)
    open(os.path.join(model_dir, "model_final.pth"), "wb").close()
    if with_config:
        with open(os.path.join(model_dir, "config.yaml"), "w") as f:
            f.write("MODEL:\n  MASK_ON: false\n")
    if with_class_names:
        with open(os.path.join(model_dir, "class_names.json"), "w") as f:
            json.dump(["dog", "cat"], f)
    return model_dir


# --- availability checks -------------------------------------------------- #
def test_availability_checks_return_bool():
    for fn in (is_onnx_available, is_onnxruntime_available, is_tensorrt_available):
        assert isinstance(fn(), bool)


# --- resolve_model_dir ----------------------------------------------------- #
def test_resolve_model_dir_valid(temp_dir):
    model_dir = _make_model_dir(temp_dir)
    resolved = resolve_model_dir(os.path.join(model_dir, "model_final.pth"))
    assert resolved["model_dir"] == model_dir
    assert resolved["config_path"].endswith("config.yaml")
    assert resolved["class_names_path"].endswith("class_names.json")


def test_resolve_model_dir_missing_weights(temp_dir):
    with pytest.raises(FileNotFoundError):
        resolve_model_dir(os.path.join(temp_dir, "nope.pth"))


def test_resolve_model_dir_missing_config(temp_dir):
    model_dir = _make_model_dir(temp_dir, with_config=False)
    with pytest.raises(FileNotFoundError):
        resolve_model_dir(os.path.join(model_dir, "model_final.pth"))


def test_resolve_model_dir_no_class_names(temp_dir):
    model_dir = _make_model_dir(temp_dir, with_class_names=False)
    resolved = resolve_model_dir(os.path.join(model_dir, "model_final.pth"))
    assert resolved["class_names_path"] is None


# --- read_class_names ------------------------------------------------------ #
def test_read_class_names(temp_dir):
    model_dir = _make_model_dir(temp_dir)
    assert read_class_names(model_dir) == ["dog", "cat"]


def test_read_class_names_absent(temp_dir):
    model_dir = _make_model_dir(temp_dir, with_class_names=False)
    assert read_class_names(model_dir) == []


def test_read_class_names_invalid(temp_dir):
    model_dir = _make_model_dir(temp_dir, with_class_names=False)
    with open(os.path.join(model_dir, "class_names.json"), "w") as f:
        f.write("not json")
    assert read_class_names(model_dir) == []


# --- copy_sidecar_files ---------------------------------------------------- #
def test_copy_sidecar_files(temp_dir):
    model_dir = _make_model_dir(temp_dir)
    out_dir = os.path.join(temp_dir, "out")
    copied = copy_sidecar_files(model_dir, out_dir)
    assert os.path.exists(os.path.join(out_dir, "config.yaml"))
    assert os.path.exists(os.path.join(out_dir, "class_names.json"))
    assert len(copied) == 2


def test_copy_sidecar_files_partial(temp_dir):
    model_dir = _make_model_dir(temp_dir, with_class_names=False)
    out_dir = os.path.join(temp_dir, "out")
    copied = copy_sidecar_files(model_dir, out_dir)
    assert os.path.exists(os.path.join(out_dir, "config.yaml"))
    assert not os.path.exists(os.path.join(out_dir, "class_names.json"))
    assert len(copied) == 1


# --- build_export_metadata ------------------------------------------------- #
def test_build_export_metadata():
    meta = build_export_metadata(
        model_format="onnx",
        opset=17,
        input_size=(800, 800),
        fp16=False,
        dynamic_axes=True,
        task="detection",
        class_names=["dog", "cat"],
        output_names=["pred_boxes", "scores", "pred_classes"],
        output_roles=["boxes", "scores", "classes"],
        preprocessing={"pixel_mean": [1, 2, 3], "pixel_std": [1, 1, 1], "input_format": "BGR"},
        timestamp="2026-01-01T00:00:00+00:00",
    )
    assert meta["format"] == "onnx"
    assert meta["opset"] == 17
    assert meta["input_size"] == [800, 800]
    assert meta["fp16"] is False
    assert meta["dynamic_axes"] is True
    assert meta["task"] == "detection"
    assert meta["class_names"] == ["dog", "cat"]
    assert meta["input_name"] == "image"
    assert meta["output_names"] == ["pred_boxes", "scores", "pred_classes"]
    assert meta["output_roles"] == ["boxes", "scores", "classes"]
    assert meta["preprocessing"]["input_format"] == "BGR"
    assert meta["created"] == "2026-01-01T00:00:00+00:00"
    assert "alchemydetect_version" in meta
    # Must be JSON-serializable
    json.dumps(meta)


def test_build_export_metadata_default_timestamp():
    meta = build_export_metadata(
        model_format="onnx",
        opset=17,
        input_size=(640, 640),
        fp16=True,
        dynamic_axes=False,
        task="instance_segmentation",
        class_names=[],
        output_names=["pred_boxes"],
        output_roles=["boxes"],
        preprocessing={},
    )
    assert isinstance(meta["created"], str) and meta["created"]


# --- detect_task_from_config (needs PyYAML) -------------------------------- #
def test_detect_task_from_config(temp_dir):
    pytest.importorskip("yaml")
    seg = os.path.join(temp_dir, "seg.yaml")
    with open(seg, "w") as f:
        f.write("MODEL:\n  MASK_ON: true\n")
    assert detect_task_from_config(seg) == "instance_segmentation"

    det = os.path.join(temp_dir, "det.yaml")
    with open(det, "w") as f:
        f.write("MODEL:\n  MASK_ON: false\n")
    assert detect_task_from_config(det) == "detection"

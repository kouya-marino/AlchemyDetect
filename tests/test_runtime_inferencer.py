"""Tests for the ONNX runtime inferencer (pure helpers, no onnxruntime needed)."""

import pytest

from alchemydetect.core.runtime_inferencer import _RuntimeInferencer, compute_resize_scale


def _bare_inferencer():
    """A base inferencer without running __init__ (no ONNX session needed)."""
    return _RuntimeInferencer.__new__(_RuntimeInferencer)


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


# --- _map_roles (pure dict logic, no heavy deps) -------------------------- #
def test_map_roles_by_name_excludes_ignore():
    obj = _bare_inferencer()
    obj._role_by_name = {"pred_boxes": "boxes", "scores": "scores", "pred_classes": "classes", "output_3": "ignore"}
    obj._output_roles = ["boxes", "scores", "classes", "ignore"]
    ordered = [("pred_boxes", "B"), ("scores", "S"), ("pred_classes", "C"), ("output_3", "X")]
    assert obj._map_roles(ordered) == {"boxes": "B", "scores": "S", "classes": "C"}


def test_map_roles_positional_fallback():
    obj = _bare_inferencer()
    obj._role_by_name = {}  # names not recognized -> fall back to positional roles
    obj._output_roles = ["boxes", "scores", "classes"]
    ordered = [("a", "B"), ("b", "S"), ("c", "C")]
    assert obj._map_roles(ordered) == {"boxes": "B", "scores": "S", "classes": "C"}


# --- _build_instances (needs numpy + torch + detectron2) ------------------ #
def test_build_instances_threshold_and_fields():
    np = pytest.importorskip("numpy")
    pytest.importorskip("detectron2")
    obj = _bare_inferencer()
    boxes = np.array([[10, 10, 50, 50], [0, 0, 20, 20]], dtype=np.float32)
    scores = np.array([0.9, 0.4], dtype=np.float32)
    classes = np.array([1, 0])
    inst = obj._build_instances((100, 200), boxes, scores, classes, None, scale=1.0, threshold=0.5)
    assert len(inst) == 1  # the 0.4 detection is dropped
    assert int(inst.pred_classes[0]) == 1
    assert float(inst.scores[0]) == pytest.approx(0.9)
    assert tuple(inst.image_size) == (100, 200)


def test_build_instances_scales_boxes_back():
    np = pytest.importorskip("numpy")
    pytest.importorskip("detectron2")
    obj = _bare_inferencer()
    boxes = np.array([[20, 20, 80, 80]], dtype=np.float32)  # resized-space coords
    inst = obj._build_instances(
        (1000, 1000), boxes, np.array([0.9], dtype=np.float32), np.array([0]), None, scale=2.0, threshold=0.5
    )
    assert inst.pred_boxes.tensor.numpy()[0].tolist() == pytest.approx([10.0, 10.0, 40.0, 40.0])


def test_build_instances_recovers_from_length_mismatch():
    # Simulates a pre-fix export where the image-size tensor leaked into "classes":
    # classes has length 2 but there is 1 box. Must not crash on the boolean mask.
    np = pytest.importorskip("numpy")
    pytest.importorskip("detectron2")
    obj = _bare_inferencer()
    boxes = np.array([[1, 1, 2, 2]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    classes = np.array([800, 800])  # wrong length
    inst = obj._build_instances((50, 50), boxes, scores, classes, None, scale=1.0, threshold=0.5)
    assert len(inst) == 1
    assert int(inst.pred_classes[0]) == 0  # reset to a safe default


def test_build_instances_empty():
    pytest.importorskip("numpy")
    pytest.importorskip("detectron2")
    obj = _bare_inferencer()
    inst = obj._build_instances((10, 10), None, None, None, None, scale=1.0, threshold=0.5)
    assert len(inst) == 0


# --- _preprocess (needs cv2 + numpy; no onnxruntime/detectron2) ----------- #
def _preproc_obj(input_format="BGR", min_size=800, max_size=1333):
    obj = _bare_inferencer()
    obj._input_format = input_format
    obj._min_size = min_size
    obj._max_size = max_size
    return obj


def test_preprocess_shape_dtype_and_scale():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    obj = _preproc_obj(min_size=800, max_size=1333)
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    chw, scale = obj._preprocess(img)
    assert scale == pytest.approx(compute_resize_scale(100, 200, 800, 1333))
    new_h, new_w = round(100 * scale), round(200 * scale)
    assert chw.shape == (3, new_h, new_w)
    assert chw.dtype == np.float32
    assert chw.flags["C_CONTIGUOUS"]


def test_preprocess_rgb_swaps_channels():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[:, :, 0] = 10  # B
    img[:, :, 2] = 30  # R
    # BGR: no swap -> channel 0 stays B (10)
    chw_bgr, _ = _preproc_obj("BGR")._preprocess(img)
    assert chw_bgr[0].mean() == pytest.approx(10.0, abs=1.0)
    # RGB: swapped -> channel 0 becomes R (30)
    chw_rgb, _ = _preproc_obj("RGB")._preprocess(img)
    assert chw_rgb[0].mean() == pytest.approx(30.0, abs=1.0)

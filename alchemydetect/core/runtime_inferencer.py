"""Run inference with exported ONNX models via onnxruntime.

The exported graph (see ``exporter.export_onnx``) takes a single raw image
tensor in the model's input format (BGR/RGB, 0-255, CHW) and applies
normalization internally. It was traced with ``do_postprocess=False``, so its
box outputs are in the resized model-input coordinate space — this module
reproduces Detectron2's test-time resize, then scales boxes back to the original
image and assembles a Detectron2 ``Instances`` object so the existing visualizer
and detections table can be reused unchanged.

Heavy imports (onnxruntime, torch, detectron2) are done lazily inside methods so
the pure ``compute_resize_scale`` helper stays importable for tests.
"""


def compute_resize_scale(height, width, min_size, max_size):
    """Return the ResizeShortestEdge scale factor for an image.

    Matches Detectron2's test-time policy: scale the shortest edge up to
    ``min_size``, but clamp so the longest edge does not exceed ``max_size``.
    """
    scale = min_size / min(height, width)
    if max(height, width) * scale > max_size:
        scale = max_size / max(height, width)
    return scale


class OnnxRuntimeInferencer:
    """Loads an exported ONNX model and runs detection/segmentation inference."""

    def __init__(self, onnx_path, metadata):
        """
        Args:
            onnx_path: Path to the exported model.onnx.
            metadata: Parsed export_metadata.json dict (preprocessing params,
                output roles, task). Falls back to sane defaults if keys are absent.
        """
        import onnxruntime as ort

        self._metadata = metadata or {}
        preprocessing = self._metadata.get("preprocessing", {})
        self._input_format = preprocessing.get("input_format", "BGR")
        self._min_size = int(preprocessing.get("min_size", 800))
        self._max_size = int(preprocessing.get("max_size", 1333))
        self._output_roles = self._metadata.get("output_roles", [])
        self.task = self._metadata.get("task", "detection")

        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    def _preprocess(self, image_bgr):
        """Resize (shortest-edge) and format an image into a CHW float32 array."""
        import cv2
        import numpy as np

        height, width = image_bgr.shape[:2]
        scale = compute_resize_scale(height, width, self._min_size, self._max_size)
        new_w, new_h = int(round(width * scale)), int(round(height * scale))
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if self._input_format == "RGB":
            resized = resized[:, :, ::-1]
        chw = np.ascontiguousarray(resized.transpose(2, 0, 1).astype(np.float32))
        return chw, scale

    def infer(self, image_bgr, threshold):
        """Run inference on a BGR image and return a Detectron2 Instances (CPU).

        Args:
            image_bgr: Original image as a BGR numpy array.
            threshold: Confidence threshold; detections below it are dropped.
        """
        chw, scale = self._preprocess(image_bgr)
        outputs = self._session.run(None, {self._input_name: chw})

        by_role = {}
        for role, array in zip(self._output_roles, outputs):
            by_role[role] = array

        return self._build_instances(
            orig_hw=image_bgr.shape[:2],
            boxes=by_role.get("boxes"),
            scores=by_role.get("scores"),
            classes=by_role.get("classes"),
            masks=by_role.get("masks"),
            scale=scale,
            threshold=threshold,
        )

    def _build_instances(self, orig_hw, boxes, scores, classes, masks, scale, threshold):
        import numpy as np
        import torch
        from detectron2.structures import Boxes, Instances

        height, width = orig_hw

        if boxes is None:
            boxes = np.zeros((0, 4), dtype=np.float32)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        n = len(boxes)

        if scores is None:
            scores = np.zeros((n,), dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        if classes is None:
            classes = np.zeros((n,), dtype=np.int64)
        classes = np.asarray(classes).reshape(-1).astype(np.int64)

        keep = scores >= threshold
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

        # Boxes are in resized coordinates; scale back to the original image.
        if scale != 0:
            boxes = boxes / scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)

        instances = Instances((height, width))
        instances.pred_boxes = Boxes(torch.from_numpy(np.ascontiguousarray(boxes)))
        instances.scores = torch.from_numpy(np.ascontiguousarray(scores))
        instances.pred_classes = torch.from_numpy(np.ascontiguousarray(classes))

        if masks is not None and len(boxes) > 0:
            self._attach_masks(instances, masks, keep, orig_hw)

        return instances

    def _attach_masks(self, instances, masks, keep, orig_hw):
        """Best-effort paste of ROI mask logits to full-image masks (experimental)."""
        try:
            import numpy as np
            import torch
            from detectron2.layers import paste_masks_in_image

            masks = np.asarray(masks, dtype=np.float32)
            if masks.ndim == 4:  # (N, 1, Hm, Wm) -> (N, Hm, Wm)
                masks = masks[:, 0, :, :]
            masks_t = torch.from_numpy(np.ascontiguousarray(masks))[keep]
            pasted = paste_masks_in_image(masks_t, instances.pred_boxes, orig_hw, threshold=0.5)
            instances.pred_masks = pasted
        except Exception:
            # Segmentation export/runtime is experimental; fall back to boxes only.
            pass

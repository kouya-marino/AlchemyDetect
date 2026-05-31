"""Run inference with exported models (ONNX via onnxruntime, or TensorRT engines).

The exported graph (see ``exporter.export_onnx``) takes a single raw image
tensor in the model's input format (BGR/RGB, 0-255, CHW) and applies
normalization internally. It was traced with ``do_postprocess=False``, so its
box outputs are in the resized model-input coordinate space — this module
reproduces Detectron2's test-time resize, then scales boxes back to the original
image and assembles a Detectron2 ``Instances`` object so the existing visualizer
and detections table can be reused unchanged.

Heavy imports (onnxruntime, tensorrt, pycuda, torch, detectron2) are done lazily
inside methods so the pure ``compute_resize_scale`` helper stays importable for
tests on machines without any of them.
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


class _RuntimeInferencer:
    """Shared preprocessing / postprocessing for exported-model runtimes.

    Subclasses implement ``_run(chw) -> list[(output_name, ndarray)]``.
    """

    def __init__(self, metadata):
        self._metadata = metadata or {}
        preprocessing = self._metadata.get("preprocessing", {})
        self._input_format = preprocessing.get("input_format", "BGR")
        self._min_size = int(preprocessing.get("min_size", 800))
        self._max_size = int(preprocessing.get("max_size", 1333))
        self._output_roles = self._metadata.get("output_roles", [])
        output_names = self._metadata.get("output_names", [])
        self._role_by_name = dict(zip(output_names, self._output_roles))
        self.task = self._metadata.get("task", "detection")
        self.active_provider = "CPU"

    # --- preprocessing -------------------------------------------------- #
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

    # --- inference ------------------------------------------------------ #
    def infer(self, image_bgr, threshold):
        """Run inference on a BGR image and return a Detectron2 Instances (CPU)."""
        chw, scale = self._preprocess(image_bgr)
        ordered = self._run(chw)
        by_role = self._map_roles(ordered)
        return self._build_instances(
            orig_hw=image_bgr.shape[:2],
            boxes=by_role.get("boxes"),
            scores=by_role.get("scores"),
            classes=by_role.get("classes"),
            masks=by_role.get("masks"),
            scale=scale,
            threshold=threshold,
        )

    def _run(self, chw):
        raise NotImplementedError

    def _map_roles(self, ordered):
        """Map ordered (name, array) outputs to semantic roles.

        Prefers the name->role mapping recorded in metadata; falls back to the
        positional role order if a tensor name isn't recognized.
        """
        by_role = {}
        for i, (name, array) in enumerate(ordered):
            role = self._role_by_name.get(name)
            if role is None and i < len(self._output_roles):
                role = self._output_roles[i]
            if role and role != "ignore":
                by_role[role] = array
        return by_role

    # --- postprocessing ------------------------------------------------- #
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

        # Safety net for models exported before the output-role mapping fix: if a
        # per-detection array's length disagrees with the box count (e.g. the
        # image-size tensor leaked into the classes role), reset it rather than
        # crash on the boolean mask. Correct results require re-exporting.
        if len(scores) != n or len(classes) != n:
            from alchemydetect.core.app_logging import get_logger

            get_logger().warning(
                "Output length mismatch (boxes=%d, scores=%d, classes=%d); this model was likely "
                "exported before the output-role-mapping fix. Please re-export it. Substituting "
                "defaults to avoid a crash.",
                n,
                len(scores),
                len(classes),
            )
            if len(scores) != n:
                scores = np.zeros((n,), dtype=np.float32)
            if len(classes) != n:
                classes = np.zeros((n,), dtype=np.int64)

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


class OnnxRuntimeInferencer(_RuntimeInferencer):
    """Runs an exported ONNX model with onnxruntime."""

    def __init__(self, onnx_path, metadata):
        super().__init__(metadata)
        self.active_provider = "CPUExecutionProvider"
        self._session = self._create_session(onnx_path)
        self._input_name = self._session.get_inputs()[0].name

    def _create_session(self, onnx_path):
        """Create the onnxruntime session, preferring CUDA but falling back to CPU.

        onnxruntime lists ``CUDAExecutionProvider`` whenever the onnxruntime-gpu
        package is installed, even when the CUDA/cuDNN runtime DLLs are missing.
        Requesting it then logs a noisy error before silently falling back. We
        attempt CUDA with the logger quieted, then read back the provider that
        actually activated so callers can report it honestly.
        """
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            ort.set_default_logger_severity(4)  # silence the expected CUDA-load failure
            try:
                session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            finally:
                ort.set_default_logger_severity(2)  # restore default (WARNING)
            active = session.get_providers()
            self.active_provider = active[0] if active else "CPUExecutionProvider"
            return session

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.active_provider = "CPUExecutionProvider"
        return session

    def _run(self, chw):
        outputs = self._session.run(None, {self._input_name: chw})
        names = [o.name for o in self._session.get_outputs()]
        return list(zip(names, outputs))


class TensorRTInferencer(_RuntimeInferencer):
    """Runs an exported TensorRT engine (requires tensorrt + pycuda; experimental)."""

    def __init__(self, engine_path, metadata):
        super().__init__(metadata)
        import pycuda.autoinit  # noqa: F401  (initializes a CUDA context)
        import pycuda.driver as cuda
        import tensorrt as trt

        self._trt = trt
        self._cuda = cuda

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(
                "Failed to deserialize the TensorRT engine. Engines are not portable "
                "across GPUs or TensorRT versions — rebuild it on this machine."
            )
        self._context = self._engine.create_execution_context()
        self._input_name = self._find_input_name()
        self.active_provider = "TensorRT"

    def _find_input_name(self):
        trt = self._trt
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                return name
        raise RuntimeError("TensorRT engine has no input tensor.")

    def _run(self, chw):
        import numpy as np

        trt = self._trt
        cuda = self._cuda

        self._context.set_input_shape(self._input_name, tuple(chw.shape))

        device_buffers = []  # keep alive until copied back
        host_outputs = []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                host = np.ascontiguousarray(chw)
                dmem = cuda.mem_alloc(host.nbytes)
                cuda.memcpy_htod(dmem, host)
                self._context.set_tensor_address(name, int(dmem))
                device_buffers.append(dmem)
            else:
                shape = tuple(self._context.get_tensor_shape(name))
                dtype = trt.nptype(self._engine.get_tensor_dtype(name))
                host = np.empty(shape, dtype=dtype)
                dmem = cuda.mem_alloc(host.nbytes)
                self._context.set_tensor_address(name, int(dmem))
                device_buffers.append(dmem)
                host_outputs.append((name, host, dmem))

        stream = cuda.Stream()
        self._context.execute_async_v3(stream.handle)
        stream.synchronize()

        ordered = []
        for name, host, dmem in host_outputs:
            cuda.memcpy_dtoh(host, dmem)
            ordered.append((name, host))
        return ordered

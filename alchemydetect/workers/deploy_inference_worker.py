"""Background thread for running inference with exported ONNX / TensorRT models."""

from PyQt6.QtCore import pyqtSignal

from alchemydetect.workers.inference_base import ImageInferenceWorker


class DeployInferenceWorker(ImageInferenceWorker):
    """Runs inference on images using an exported ONNX (.onnx) or TensorRT (.engine) model."""

    provider_ready = pyqtSignal(str)  # active runtime provider (e.g. CPUExecutionProvider)
    INIT_ERROR_LABEL = "ONNX inference initialization error"

    def __init__(self, model_path, metadata, image_paths, threshold=0.5, class_names=None):
        """
        Args:
            model_path: Path to the exported model (.onnx or .engine).
            metadata: Parsed export_metadata.json dict.
            image_paths: List of image file paths.
            threshold: Confidence threshold.
            class_names: Optional list of class name strings for the overlay.
        """
        super().__init__(image_paths, threshold, class_names)
        self._model_path = model_path
        self._export_metadata = metadata

    def _setup(self):
        from alchemydetect.core.app_logging import get_logger

        if str(self._model_path).lower().endswith(".engine"):
            from alchemydetect.core.runtime_inferencer import TensorRTInferencer

            inferencer = TensorRTInferencer(self._model_path, self._export_metadata)
        else:
            from alchemydetect.core.runtime_inferencer import OnnxRuntimeInferencer

            inferencer = OnnxRuntimeInferencer(self._model_path, self._export_metadata)
        get_logger().info("Deploy: runtime provider = %s", inferencer.active_provider)
        self.provider_ready.emit(inferencer.active_provider)

        viz_metadata = self._make_metadata("__alchemy_deploy")
        threshold = self._threshold

        def detect(image_bgr):
            return inferencer.infer(image_bgr, threshold)

        return detect, viz_metadata

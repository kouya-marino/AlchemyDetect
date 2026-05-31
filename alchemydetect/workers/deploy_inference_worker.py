"""Background thread for running inference with exported ONNX models."""

from PyQt6.QtCore import QThread, pyqtSignal

from alchemydetect.workers.inference_worker import InferenceWorker


class DeployInferenceWorker(QThread):
    """Runs inference on images using an exported ONNX model in a background thread."""

    # Signals: (image_path, instances, annotated_image_rgb)
    result_ready = pyqtSignal(str, object, object)
    progress = pyqtSignal(int, int)  # (current, total)
    error = pyqtSignal(str)
    finished_all = pyqtSignal()

    def __init__(self, onnx_path, metadata, image_paths, threshold=0.5, class_names=None):
        """
        Args:
            onnx_path: Path to the exported model.onnx.
            metadata: Parsed export_metadata.json dict.
            image_paths: List of image file paths.
            threshold: Confidence threshold.
            class_names: Optional list of class name strings for the overlay.
        """
        super().__init__()
        self._onnx_path = onnx_path
        self._metadata = metadata
        self._image_paths = image_paths
        self._threshold = threshold
        self._class_names = class_names or []
        self._should_stop = False

    def stop(self):
        self._should_stop = True

    def run(self):
        try:
            import cv2
            from detectron2.data import MetadataCatalog

            from alchemydetect.core.app_logging import get_logger
            from alchemydetect.core.inferencer import visualize_predictions
            from alchemydetect.core.runtime_inferencer import OnnxRuntimeInferencer

            inferencer = OnnxRuntimeInferencer(self._onnx_path, self._metadata)
            get_logger().info("Deploy: onnxruntime provider = %s", inferencer.active_provider)

            metadata = None
            if self._class_names:
                meta_name = "__alchemy_deploy"
                if meta_name in MetadataCatalog.list():
                    MetadataCatalog.remove(meta_name)
                MetadataCatalog.get(meta_name).set(thing_classes=self._class_names)
                metadata = MetadataCatalog.get(meta_name)

            total = len(self._image_paths)
            for i, img_path in enumerate(self._image_paths):
                if self._should_stop:
                    break
                try:
                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is None:
                        raise ValueError(f"Could not read image: {img_path}")
                    instances = inferencer.infer(img_bgr, self._threshold)
                    annotated = visualize_predictions(img_bgr, instances, metadata)
                    self.result_ready.emit(str(img_path), instances, annotated)
                except Exception as e:
                    self.error.emit(f"Error on {img_path}: {e}")

                self.progress.emit(i + 1, total)

            self.finished_all.emit()

        except Exception as e:
            self.error.emit(f"ONNX inference initialization error: {e}")

    @staticmethod
    def collect_image_paths(path):
        """Collect image file paths from a file or directory (reuses InferenceWorker)."""
        return InferenceWorker.collect_image_paths(path)

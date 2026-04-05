"""Background thread for running Detectron2 inference."""

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal


class InferenceWorker(QThread):
    """Runs inference on one or more images in a background thread."""

    # Signals: (image_path, instances, annotated_image_rgb)
    result_ready = pyqtSignal(str, object, object)
    progress = pyqtSignal(int, int)  # (current, total)
    error = pyqtSignal(str)
    finished_all = pyqtSignal()

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, config_yaml_path, weights_path, image_paths, threshold=0.5, class_names=None):
        """
        Args:
            config_yaml_path: Path to saved config.yaml.
            weights_path: Path to .pth weights file.
            image_paths: List of image file paths.
            threshold: Confidence threshold.
            class_names: Optional list of class name strings.
        """
        super().__init__()
        self._config_yaml_path = config_yaml_path
        self._weights_path = weights_path
        self._image_paths = image_paths
        self._threshold = threshold
        self._class_names = class_names or []
        self._should_stop = False

    def stop(self):
        self._should_stop = True

    def run(self):
        try:
            from alchemydetect.core.inferencer import (
                load_predictor,
                run_inference_single,
                visualize_predictions,
            )

            from detectron2.data import MetadataCatalog

            predictor, cfg = load_predictor(self._config_yaml_path, self._weights_path, self._threshold)

            # Set up metadata with class names for visualization
            metadata = None
            if self._class_names:
                meta_name = "__alchemy_inference"
                if meta_name in MetadataCatalog.list():
                    MetadataCatalog.remove(meta_name)
                MetadataCatalog.get(meta_name).set(thing_classes=self._class_names)
                metadata = MetadataCatalog.get(meta_name)

            total = len(self._image_paths)
            for i, img_path in enumerate(self._image_paths):
                if self._should_stop:
                    break

                try:
                    img_bgr, instances = run_inference_single(predictor, img_path)
                    annotated = visualize_predictions(img_bgr, instances, metadata)
                    self.result_ready.emit(str(img_path), instances, annotated)
                except Exception as e:
                    self.error.emit(f"Error on {img_path}: {e}")

                self.progress.emit(i + 1, total)

            self.finished_all.emit()

        except Exception as e:
            self.error.emit(f"Inference initialization error: {e}")

    @classmethod
    def collect_image_paths(cls, path):
        """Collect image file paths from a file or directory path.

        Args:
            path: A single image path or a directory path.

        Returns:
            List of Path objects.
        """
        p = Path(path)
        if p.is_file():
            return [p]
        elif p.is_dir():
            paths = sorted(f for f in p.iterdir() if f.suffix.lower() in cls.IMAGE_EXTENSIONS)
            return paths
        return []

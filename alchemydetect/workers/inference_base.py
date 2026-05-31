"""Base QThread that runs a per-image detect → visualize loop off the GUI thread.

Subclasses implement ``_setup()`` → ``(detect, metadata)`` where
``detect(image_bgr)`` returns a Detectron2 ``Instances`` (on CPU) and ``metadata``
is the visualizer metadata (or None). All heavy imports stay inside ``run()`` /
``_setup()`` so importing this module is cheap.
"""

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal


class ImageInferenceWorker(QThread):
    """Runs detection over a list of images and emits per-image results."""

    # Signals: (image_path, instances, annotated_image_rgb, detection_ms)
    result_ready = pyqtSignal(str, object, object, float)
    progress = pyqtSignal(int, int)  # (current, total)
    error = pyqtSignal(str)
    finished_all = pyqtSignal()

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    INIT_ERROR_LABEL = "Inference initialization error"

    def __init__(self, image_paths, threshold=0.5, class_names=None):
        super().__init__()
        self._image_paths = image_paths
        self._threshold = threshold
        self._class_names = class_names or []
        self._should_stop = False

    def stop(self):
        self._should_stop = True

    def _setup(self):
        """Return ``(detect, metadata)``. Override in subclasses (heavy imports here)."""
        raise NotImplementedError

    def _make_metadata(self, meta_name):
        """Build a MetadataCatalog entry from class_names for the visualizer, or None."""
        if not self._class_names:
            return None
        from detectron2.data import MetadataCatalog

        if meta_name in MetadataCatalog.list():
            MetadataCatalog.remove(meta_name)
        MetadataCatalog.get(meta_name).set(thing_classes=self._class_names)
        return MetadataCatalog.get(meta_name)

    def run(self):
        import time

        import cv2

        from alchemydetect.core.inferencer import visualize_predictions

        try:
            detect, metadata = self._setup()
            total = len(self._image_paths)
            for i, img_path in enumerate(self._image_paths):
                if self._should_stop:
                    break
                try:
                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is None:
                        raise ValueError(f"Could not read image: {img_path}")
                    start = time.perf_counter()
                    instances = detect(img_bgr)
                    detection_ms = (time.perf_counter() - start) * 1000.0
                    annotated = visualize_predictions(img_bgr, instances, metadata)
                    self.result_ready.emit(str(img_path), instances, annotated, detection_ms)
                except Exception as e:
                    self.error.emit(f"Error on {img_path}: {e}")
                self.progress.emit(i + 1, total)
        except Exception as e:
            self.error.emit(f"{self.INIT_ERROR_LABEL}: {e}")
        finally:
            self.finished_all.emit()

    @classmethod
    def collect_image_paths(cls, path):
        """Collect image file paths from a single file or a directory."""
        p = Path(path)
        if p.is_file():
            return [p]
        if p.is_dir():
            return sorted(f for f in p.iterdir() if f.suffix.lower() in cls.IMAGE_EXTENSIONS)
        return []

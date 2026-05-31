"""Inference tab: load a trained .pth model, run on images, display results."""

from pathlib import Path

from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from alchemydetect.workers.inference_worker import InferenceWorker

from .dialogs import browse_directory, browse_file, load_model_dialog
from .results_viewer import ResultsViewerMixin


class InferenceTab(ResultsViewerMixin, QWidget):
    """Tab for loading a trained model and running inference."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config_path = None
        self._weights_path = None
        self._class_names = []
        self._worker = None
        self._results = []  # (path, instances, annotated_rgb, detection_ms)
        self._current_idx = 0
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Model loading group ---
        model_group = QGroupBox("Model")
        mg_layout = QHBoxLayout(model_group)

        self._model_label = QLabel("No model loaded")
        mg_layout.addWidget(self._model_label)

        load_btn = QPushButton("Load Model")
        load_btn.clicked.connect(self._on_load_model)
        mg_layout.addWidget(load_btn)

        mg_layout.addWidget(QLabel("Threshold:"))
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.01, 1.0)
        self._threshold_spin.setValue(0.5)
        self._threshold_spin.setSingleStep(0.05)
        self._threshold_spin.setDecimals(2)
        mg_layout.addWidget(self._threshold_spin)

        main_layout.addWidget(model_group)

        # --- Input controls ---
        input_row = QHBoxLayout()

        self._single_btn = QPushButton("Run on Image")
        self._single_btn.setEnabled(False)
        self._single_btn.clicked.connect(self._on_run_single)
        input_row.addWidget(self._single_btn)

        self._folder_btn = QPushButton("Run on Folder")
        self._folder_btn.setEnabled(False)
        self._folder_btn.clicked.connect(self._on_run_folder)
        input_row.addWidget(self._folder_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        input_row.addWidget(self._stop_btn)

        main_layout.addLayout(input_row)

        # --- Progress bar ---
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        main_layout.addWidget(self._progress)

        # --- Results area (image viewer + detections table + nav) ---
        main_layout.addWidget(self._build_results_panel(), stretch=1)

    def _on_load_model(self):
        config_path, weights_path = load_model_dialog(self)
        if config_path and weights_path:
            self._config_path = config_path
            self._weights_path = weights_path
            name = Path(weights_path).parent.name
            self._model_label.setText(f"Loaded: {name}/{Path(weights_path).name}")
            self._single_btn.setEnabled(True)
            self._folder_btn.setEnabled(True)

            # Load class names if available
            self._class_names = []
            class_names_file = Path(weights_path).parent / "class_names.json"
            if class_names_file.exists():
                import json

                with open(class_names_file, "r") as f:
                    self._class_names = json.load(f)

    def _on_run_single(self):
        img_filter = "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;All Files (*)"
        path = browse_file(self, "Select Image", filter_str=img_filter)
        if path:
            self._start_inference([path])

    def _on_run_folder(self):
        folder = browse_directory(self, "Select Image Folder")
        if folder:
            paths = InferenceWorker.collect_image_paths(folder)
            if not paths:
                QMessageBox.warning(self, "No Images", "No supported image files found in the selected folder.")
                return
            self._start_inference([str(p) for p in paths])

    def _start_inference(self, image_paths):
        if not self._config_path or not self._weights_path:
            QMessageBox.warning(self, "No Model", "Please load a model first.")
            return

        self._results.clear()
        self._current_idx = 0
        self._image_viewer.clear_image()
        self._table.setRowCount(0)
        self._timing_label.setText("")

        self._progress.setVisible(True)
        self._progress.setMaximum(len(image_paths))
        self._progress.setValue(0)

        self._single_btn.setEnabled(False)
        self._folder_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        self._worker = InferenceWorker(
            self._config_path,
            self._weights_path,
            image_paths,
            threshold=self._threshold_spin.value(),
            class_names=self._class_names,
        )
        self._worker.result_ready.connect(self._on_result)
        self._worker.progress.connect(self._on_progress)
        self._worker.error.connect(self._on_error)
        self._worker.finished_all.connect(self._on_finished)
        self._worker.start()

    def _on_stop(self):
        if self._worker:
            self._worker.stop()

    def _on_progress(self, current, total):
        self._progress.setValue(current)

    def _on_error(self, msg):
        self._info_label.setText(f"Error: {msg}")
        from alchemydetect.core.app_logging import get_logger

        get_logger().error("Inference: %s", msg)

    def _on_finished(self):
        self._progress.setVisible(False)
        self._single_btn.setEnabled(True)
        self._folder_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._update_nav()
        # finished_all fires from the worker's run() before it fully returns; wait
        # for the thread to actually terminate before dropping our only reference,
        # or Qt aborts with "QThread: Destroyed while thread is still running".
        if self._worker is not None:
            self._worker.wait(5000)
            self._worker = None

    def shutdown(self):
        """Stop and await the worker thread (called on app close)."""
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None

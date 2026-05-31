"""Deploy tab: run inference with an exported ONNX / TensorRT model (faster than Detectron2)."""

import json
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

from alchemydetect.core.exporter import is_onnxruntime_available, is_tensorrt_available
from alchemydetect.workers.deploy_inference_worker import DeployInferenceWorker

from .dialogs import browse_directory, browse_file
from .results_viewer import ResultsViewerMixin


class DeployTab(ResultsViewerMixin, QWidget):
    """Tab for running inference with an exported ONNX (.onnx) or TensorRT (.engine) model."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model_path = None
        self._metadata = None
        self._class_names = []
        self._worker = None
        self._results = []  # (path, instances, annotated_rgb, detection_ms)
        self._current_idx = 0
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Model loading group ---
        model_group = QGroupBox("Exported Model")
        mg_layout = QHBoxLayout(model_group)

        self._model_label = QLabel("No model loaded")
        mg_layout.addWidget(self._model_label, stretch=1)

        load_btn = QPushButton("Load Model...")
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

        # --- Results area (with the runtime-provider label above the timing label) ---
        self._provider_label = QLabel("")
        self._provider_label.setStyleSheet("color: #6a1b9a; font-weight: bold;")
        main_layout.addWidget(self._build_results_panel(extra_widgets=[self._provider_label]), stretch=1)

    def _on_load_model(self):
        model_filter = "Exported Models (*.onnx *.engine);;ONNX (*.onnx);;TensorRT (*.engine);;All Files (*)"
        path = browse_file(self, "Select Exported Model", filter_str=model_filter)
        if not path:
            return

        model_dir = Path(path).parent
        meta_path = model_dir / "export_metadata.json"
        if not meta_path.exists():
            QMessageBox.warning(
                self,
                "Missing Metadata",
                "No export_metadata.json found next to the model.\n"
                "Re-export the model from the Export tab so the runtime knows how to "
                "preprocess inputs and interpret outputs.",
            )
            return

        try:
            with open(meta_path, "r") as f:
                self._metadata = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            QMessageBox.critical(self, "Metadata Error", f"Could not read export_metadata.json:\n{e}")
            return

        self._model_path = path
        self._class_names = self._metadata.get("class_names") or []
        if not self._class_names:
            cn_path = model_dir / "class_names.json"
            if cn_path.exists():
                try:
                    with open(cn_path, "r") as f:
                        self._class_names = json.load(f)
                except (json.JSONDecodeError, OSError):
                    self._class_names = []

        task = self._metadata.get("task", "?")
        self._model_label.setText(f"Loaded: {Path(path).name} ({task}, {len(self._class_names)} classes)")
        self._single_btn.setEnabled(True)
        self._folder_btn.setEnabled(True)

    def _on_run_single(self):
        img_filter = "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;All Files (*)"
        path = browse_file(self, "Select Image", filter_str=img_filter)
        if path:
            self._start_inference([path])

    def _on_run_folder(self):
        folder = browse_directory(self, "Select Image Folder")
        if folder:
            paths = DeployInferenceWorker.collect_image_paths(folder)
            if not paths:
                QMessageBox.warning(self, "No Images", "No supported image files found in the selected folder.")
                return
            self._start_inference([str(p) for p in paths])

    def _start_inference(self, image_paths):
        if not self._model_path or not self._metadata:
            QMessageBox.warning(self, "No Model", "Please load an exported model first.")
            return
        is_engine = str(self._model_path).lower().endswith(".engine")
        if is_engine and not is_tensorrt_available():
            QMessageBox.critical(
                self,
                "TensorRT Not Installed",
                "Running a TensorRT engine requires the 'tensorrt' (and 'pycuda') packages.\n\n"
                "TensorRT is not pip-installable from this project — install it manually to "
                "match your CUDA/cuDNN versions (see INSTALL.md).",
            )
            return
        if not is_engine and not is_onnxruntime_available():
            QMessageBox.critical(
                self,
                "onnxruntime Not Installed",
                "Running ONNX models requires the 'onnxruntime' package.\n\n"
                "Install the export extras with:\n    pip install alchemydetect[export]",
            )
            return

        self._results.clear()
        self._current_idx = 0
        self._image_viewer.clear_image()
        self._table.setRowCount(0)
        self._timing_label.setText("")
        self._provider_label.setText("")

        self._progress.setVisible(True)
        self._progress.setMaximum(len(image_paths))
        self._progress.setValue(0)

        self._single_btn.setEnabled(False)
        self._folder_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        self._worker = DeployInferenceWorker(
            self._model_path,
            self._metadata,
            image_paths,
            threshold=self._threshold_spin.value(),
            class_names=self._class_names,
        )
        self._worker.result_ready.connect(self._on_result)
        self._worker.progress.connect(self._on_progress)
        self._worker.error.connect(self._on_error)
        self._worker.finished_all.connect(self._on_finished)
        self._worker.provider_ready.connect(self._on_provider)
        self._worker.start()

    def _on_stop(self):
        if self._worker:
            self._worker.stop()

    def _on_provider(self, provider):
        runtime = "TensorRT" if str(self._model_path).lower().endswith(".engine") else "onnxruntime"
        self._provider_label.setText(f"Runtime: {runtime} — {provider}")

    def _on_progress(self, current, total):
        self._progress.setValue(current)

    def _on_error(self, msg):
        self._info_label.setText(f"Error: {msg}")
        from alchemydetect.core.app_logging import get_logger

        get_logger().error("Deploy: %s", msg)

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

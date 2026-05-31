"""Deploy tab: run inference with an exported ONNX model (faster than Detectron2)."""

import json
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from alchemydetect.core.exporter import is_onnxruntime_available, is_tensorrt_available
from alchemydetect.workers.deploy_inference_worker import DeployInferenceWorker

from .dialogs import browse_directory, browse_file
from .image_viewer import ImageViewer


class DeployTab(QWidget):
    """Tab for running inference with an exported ONNX model."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model_path = None
        self._metadata = None
        self._class_names = []
        self._worker = None
        self._results = []  # List of (path, instances, annotated_rgb, detection_ms)
        self._current_idx = 0
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Model loading group ---
        model_group = QGroupBox("ONNX Model")
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

        # --- Results area (image viewer + detections table) ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._image_viewer = ImageViewer()
        splitter.addWidget(self._image_viewer)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self._info_label = QLabel("")
        right_layout.addWidget(self._info_label)

        self._provider_label = QLabel("")
        self._provider_label.setStyleSheet("color: #6a1b9a; font-weight: bold;")
        right_layout.addWidget(self._provider_label)

        self._timing_label = QLabel("")
        self._timing_label.setStyleSheet("color: #1565c0; font-weight: bold;")
        right_layout.addWidget(self._timing_label)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Class", "Score", "BBox"])
        self._table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self._table)

        nav_row = QHBoxLayout()
        self._prev_btn = QPushButton("< Prev")
        self._prev_btn.setEnabled(False)
        self._prev_btn.clicked.connect(self._on_prev)
        nav_row.addWidget(self._prev_btn)

        self._nav_label = QLabel("")
        self._nav_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_row.addWidget(self._nav_label)

        self._next_btn = QPushButton("Next >")
        self._next_btn.setEnabled(False)
        self._next_btn.clicked.connect(self._on_next)
        nav_row.addWidget(self._next_btn)

        right_layout.addLayout(nav_row)
        splitter.addWidget(right_panel)
        splitter.setSizes([700, 300])

        main_layout.addWidget(splitter, stretch=1)

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

    def _on_result(self, image_path, instances, annotated_rgb, detection_ms):
        self._results.append((image_path, instances, annotated_rgb, detection_ms))
        if len(self._results) == 1:
            self._show_result(0)

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
        self._worker = None

    def _show_result(self, idx):
        if idx < 0 or idx >= len(self._results):
            return
        self._current_idx = idx
        path, instances, annotated_rgb, detection_ms = self._results[idx]

        self._image_viewer.set_image_rgb(annotated_rgb)
        self._info_label.setText(f"{Path(path).name} — {len(instances)} detections")
        timing = f"Detection time: {detection_ms:.1f} ms"
        if detection_ms > 0:
            timing += f" ({1000.0 / detection_ms:.1f} FPS)"
        self._timing_label.setText(timing)

        self._table.setRowCount(len(instances))
        boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
        scores = instances.scores.numpy() if instances.has("scores") else []
        classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []

        for i in range(len(instances)):
            cls_id = int(classes[i]) if i < len(classes) else -1
            if self._class_names and 0 <= cls_id < len(self._class_names):
                cls_text = self._class_names[cls_id]
            else:
                cls_text = str(cls_id) if cls_id >= 0 else "?"
            score_text = f"{scores[i]:.3f}" if i < len(scores) else "?"
            bbox_text = ""
            if i < len(boxes):
                b = boxes[i]
                bbox_text = f"[{b[0]:.0f}, {b[1]:.0f}, {b[2]:.0f}, {b[3]:.0f}]"

            self._table.setItem(i, 0, QTableWidgetItem(cls_text))
            self._table.setItem(i, 1, QTableWidgetItem(score_text))
            self._table.setItem(i, 2, QTableWidgetItem(bbox_text))

        self._update_nav()

    def _update_nav(self):
        total = len(self._results)
        self._prev_btn.setEnabled(self._current_idx > 0)
        self._next_btn.setEnabled(self._current_idx < total - 1)
        if total > 0:
            self._nav_label.setText(f"{self._current_idx + 1} / {total}")
        else:
            self._nav_label.setText("")

    def _on_prev(self):
        self._show_result(self._current_idx - 1)

    def _on_next(self):
        self._show_result(self._current_idx + 1)

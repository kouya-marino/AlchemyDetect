"""Inference tab: load model, run on images, display results."""

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

from alchemydetect.workers.inference_worker import InferenceWorker

from .dialogs import browse_directory, browse_file, load_model_dialog
from .image_viewer import ImageViewer


class InferenceTab(QWidget):
    """Tab for loading a trained model and running inference."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config_path = None
        self._weights_path = None
        self._class_names = []
        self._worker = None
        self._results = []  # List of (path, instances, annotated_rgb)
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

        # --- Results area (image viewer + detections table) ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._image_viewer = ImageViewer()
        splitter.addWidget(self._image_viewer)

        # Right side: detections table + navigation
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self._info_label = QLabel("")
        right_layout.addWidget(self._info_label)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Class", "Score", "BBox"])
        self._table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self._table)

        # Navigation
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

    def _on_result(self, image_path, instances, annotated_rgb):
        self._results.append((image_path, instances, annotated_rgb))
        # Show the first result immediately, then update nav
        if len(self._results) == 1:
            self._show_result(0)

    def _on_progress(self, current, total):
        self._progress.setValue(current)

    def _on_error(self, msg):
        self._info_label.setText(f"Error: {msg}")

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
        path, instances, annotated_rgb = self._results[idx]

        self._image_viewer.set_image_rgb(annotated_rgb)
        self._info_label.setText(f"{Path(path).name} — {len(instances)} detections")

        # Populate detections table
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

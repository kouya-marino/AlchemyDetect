"""Shared results panel + navigation for inference-style tabs.

Mixed into tabs that display per-image detection results (Inference, Deploy). The
host QWidget calls ``_build_results_panel()`` while building its UI and maintains
``self._results`` (list of ``(path, instances, annotated_rgb, detection_ms)``),
``self._current_idx`` and ``self._class_names``.
"""

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .image_viewer import ImageViewer


class ResultsViewerMixin:
    """Image viewer + detections table + prev/next navigation, shared across tabs."""

    def _build_results_panel(self, extra_widgets=()):
        """Build and return the results splitter (image viewer | table + nav).

        Args:
            extra_widgets: optional widgets to insert between the info label and the
                timing label (e.g. the Deploy tab's runtime-provider label).
        """
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._image_viewer = ImageViewer()
        splitter.addWidget(self._image_viewer)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self._info_label = QLabel("")
        right_layout.addWidget(self._info_label)

        for widget in extra_widgets:
            right_layout.addWidget(widget)

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
        return splitter

    def _on_result(self, image_path, instances, annotated_rgb, detection_ms):
        self._results.append((image_path, instances, annotated_rgb, detection_ms))
        # Show the first result immediately, then update nav as more arrive.
        if len(self._results) == 1:
            self._show_result(0)

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
        self._nav_label.setText(f"{self._current_idx + 1} / {total}" if total > 0 else "")

    def _on_prev(self):
        self._show_result(self._current_idx - 1)

    def _on_next(self):
        self._show_result(self._current_idx + 1)

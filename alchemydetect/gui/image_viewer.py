"""Image viewer widget that displays images with detection overlays."""

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget


class ImageViewer(QWidget):
    """Scrollable image display widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        scroll = QScrollArea()
        scroll.setWidget(self._label)
        scroll.setWidgetResizable(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

        self._current_pixmap = None

    def set_image_rgb(self, image_rgb):
        """Display an RGB numpy array image.

        Args:
            image_rgb: numpy array of shape (H, W, 3) with dtype uint8.
        """
        if image_rgb is None:
            self._label.clear()
            self._current_pixmap = None
            return

        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._current_pixmap = QPixmap.fromImage(qimg)
        self._update_display()

    def _update_display(self):
        """Scale pixmap to fit the label while keeping aspect ratio."""
        if self._current_pixmap is None:
            return
        scaled = self._current_pixmap.scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def clear_image(self):
        """Clear the displayed image."""
        self._label.clear()
        self._current_pixmap = None

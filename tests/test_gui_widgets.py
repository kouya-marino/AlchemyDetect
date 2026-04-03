"""Tests for GUI widget modules (non-interactive)."""

import numpy as np
from PyQt6.QtWidgets import QApplication

# Ensure QApplication exists for widget tests
app = QApplication.instance() or QApplication([])


def test_log_viewer():
    from alchemydetect.gui.log_viewer import LogViewer

    viewer = LogViewer()
    viewer.append_log("Test message 1")
    viewer.append_log("Test message 2")
    text = viewer.toPlainText()
    assert "Test message 1" in text
    assert "Test message 2" in text

    viewer.clear_logs()
    assert viewer.toPlainText() == ""


def test_loss_plot():
    from alchemydetect.gui.loss_plot import LossPlot

    plot = LossPlot()
    plot.add_point(1, 2.5)
    plot.add_point(2, 1.8)
    assert plot._iterations == [1, 2]
    assert plot._losses == [2.5, 1.8]

    plot.clear_plot()
    assert plot._iterations == []
    assert plot._losses == []


def test_image_viewer():
    from alchemydetect.gui.image_viewer import ImageViewer

    viewer = ImageViewer()
    img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    viewer.set_image_rgb(img)
    assert viewer._current_pixmap is not None

    viewer.clear_image()
    assert viewer._current_pixmap is None


def test_image_viewer_none_image():
    from alchemydetect.gui.image_viewer import ImageViewer

    viewer = ImageViewer()
    viewer.set_image_rgb(None)
    assert viewer._current_pixmap is None

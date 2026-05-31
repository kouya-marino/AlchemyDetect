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


def test_train_tab_progress_bar():
    from alchemydetect.gui.train_tab import TrainTab

    tab = TrainTab()
    # Progress bar exists and is hidden until training starts
    assert tab._progress is not None
    assert not tab._progress.isVisible()


def test_export_tab():
    from alchemydetect.core.exporter import is_onnxruntime_available
    from alchemydetect.gui.export_tab import ExportTab

    tab = ExportTab()
    # Format combo populated with ONNX
    assert tab._format_combo.count() >= 1
    assert tab._format_combo.itemText(0) == "ONNX"
    # Sensible option defaults
    assert tab._opset_spin.value() == 17
    assert tab._height_spin.value() == 800
    assert tab._width_spin.value() == 800
    # No model loaded initially; export guarded
    assert tab._resolved is None
    # Validate checkbox reflects onnxruntime availability
    assert tab._validate_check.isEnabled() == is_onnxruntime_available()


def test_inference_tab():
    from alchemydetect.gui.inference_tab import InferenceTab

    tab = InferenceTab()
    # Shared results panel built; no model loaded; run buttons disabled
    assert tab._info_label is not None
    assert tab._table.columnCount() == 3
    assert not tab._single_btn.isEnabled()
    assert tab._threshold_spin.value() == 0.5


def test_deploy_tab():
    from alchemydetect.gui.deploy_tab import DeployTab

    tab = DeployTab()
    # Provider label is Deploy-specific; shared panel still present
    assert tab._provider_label is not None
    assert tab._table.columnCount() == 3
    # No model loaded initially; run buttons disabled; defaults
    assert tab._model_path is None
    assert not tab._single_btn.isEnabled()
    assert not tab._folder_btn.isEnabled()
    assert tab._threshold_spin.value() == 0.5
    assert tab._timing_label.text() == ""


def test_results_viewer_update_nav():
    # Shared ResultsViewerMixin navigation logic (prev/next enabled + label).
    from alchemydetect.gui.inference_tab import InferenceTab

    tab = InferenceTab()
    tab._results = [object(), object(), object()]  # _update_nav only uses the count

    tab._current_idx = 1
    tab._update_nav()
    assert tab._prev_btn.isEnabled() and tab._next_btn.isEnabled()
    assert tab._nav_label.text() == "2 / 3"

    tab._current_idx = 0
    tab._update_nav()
    assert not tab._prev_btn.isEnabled() and tab._next_btn.isEnabled()

    tab._current_idx = 2
    tab._update_nav()
    assert tab._prev_btn.isEnabled() and not tab._next_btn.isEnabled()

    tab._results = []
    tab._update_nav()
    assert tab._nav_label.text() == ""

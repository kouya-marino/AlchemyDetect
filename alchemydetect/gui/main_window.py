"""Main application window."""

from PyQt6.QtWidgets import QMainWindow, QTabWidget

from .deploy_tab import DeployTab
from .export_tab import ExportTab
from .inference_tab import InferenceTab
from .train_tab import TrainTab


class MainWindow(QMainWindow):
    """AlchemyDetect main window with Train and Inference tabs."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AlchemyDetect — Detectron2 Training & Inference")
        self.setMinimumSize(1100, 750)

        # Tab widget
        tabs = QTabWidget()
        self._train_tab = TrainTab()
        self._inference_tab = InferenceTab()
        self._export_tab = ExportTab()
        self._deploy_tab = DeployTab()

        tabs.addTab(self._train_tab, "Train")
        tabs.addTab(self._inference_tab, "Inference")
        tabs.addTab(self._export_tab, "Export")
        tabs.addTab(self._deploy_tab, "Deploy")

        self.setCentralWidget(tabs)

    def closeEvent(self, event):
        """Stop background work cleanly so we don't orphan the training process or
        destroy a still-running QThread on exit."""
        for tab in (self._train_tab, self._inference_tab, self._export_tab, self._deploy_tab):
            shutdown = getattr(tab, "shutdown", None)
            if shutdown is not None:
                try:
                    shutdown()
                except Exception:
                    pass
        super().closeEvent(event)

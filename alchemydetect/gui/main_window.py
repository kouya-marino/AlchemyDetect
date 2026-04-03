"""Main application window."""

from PyQt6.QtWidgets import QMainWindow, QTabWidget

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

        tabs.addTab(self._train_tab, "Train")
        tabs.addTab(self._inference_tab, "Inference")

        self.setCentralWidget(tabs)

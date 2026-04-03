"""AlchemyDetect — Detectron2 Training & Inference GUI."""

import sys
import multiprocessing

from PyQt6.QtWidgets import QApplication

from alchemydetect.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AlchemyDetect")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required for Windows multiprocessing
    main()

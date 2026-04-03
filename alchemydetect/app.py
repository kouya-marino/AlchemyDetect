"""Entry point for the AlchemyDetect application."""

import multiprocessing
import sys

from PyQt6.QtWidgets import QApplication

from alchemydetect.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AlchemyDetect")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

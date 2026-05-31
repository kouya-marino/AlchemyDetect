"""Entry point for the AlchemyDetect application."""

import multiprocessing
import sys

from PyQt6.QtWidgets import QApplication

from alchemydetect.core.app_logging import init_logging
from alchemydetect.gui.main_window import MainWindow


def main():
    # Safe for the gui-script entry point too (which bypasses __main__); required
    # for Windows multiprocessing/spawn in frozen builds.
    multiprocessing.freeze_support()
    log_path = init_logging()

    app = QApplication(sys.argv)
    app.setApplicationName("AlchemyDetect")

    window = MainWindow()
    window.show()
    window.statusBar().showMessage(f"Logging to {log_path}")

    sys.exit(app.exec())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

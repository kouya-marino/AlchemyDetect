"""Live log viewer widget."""

from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QPlainTextEdit

from alchemydetect.core.app_logging import get_logger


class LogViewer(QPlainTextEdit):
    """Read-only text widget for displaying live training logs.

    Appended lines are also mirrored to the session log file (if logging has
    been initialized) so they can be analyzed after the app closes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)  # Limit memory usage

    def append_log(self, text):
        """Append a log line, auto-scroll, and mirror it to the session log file."""
        self.appendPlainText(text)
        self.moveCursor(QTextCursor.MoveOperation.End)
        get_logger().info(text)

    def clear_logs(self):
        """Clear all log content."""
        self.clear()

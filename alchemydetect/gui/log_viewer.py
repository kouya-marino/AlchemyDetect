"""Live log viewer widget."""

from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QPlainTextEdit


class LogViewer(QPlainTextEdit):
    """Read-only text widget for displaying live training logs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)  # Limit memory usage

    def append_log(self, text):
        """Append a log line and auto-scroll to bottom."""
        self.appendPlainText(text)
        self.moveCursor(QTextCursor.MoveOperation.End)

    def clear_logs(self):
        """Clear all log content."""
        self.clear()

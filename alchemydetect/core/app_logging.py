"""Persistent file logging into a ``logs/`` directory for post-hoc analysis.

Everything shown in the GUI log views (training and export progress, including
relayed worker tracebacks) and inference errors are mirrored to a timestamped
file under the logs directory so issues can be analyzed after the fact.

The logs directory is ``$ALCHEMYDETECT_LOG_DIR`` if set, otherwise ``logs/`` in
the current working directory (the app's folder when launched via
``python main.py``).
"""

import logging
import os
from datetime import datetime
from pathlib import Path

LOGGER_NAME = "alchemydetect"
_SESSION_MARKER = "_alchemy_session"


MAX_LOG_FILES = 20


def get_log_dir():
    """Return (creating if needed) the directory where log files are written."""
    override = os.environ.get("ALCHEMYDETECT_LOG_DIR")
    log_dir = Path(override) if override else Path.cwd() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def prune_old_logs(log_dir, keep=MAX_LOG_FILES):
    """Delete all but the newest `keep` session log files (best-effort).

    Filenames are timestamped (alchemydetect_YYYYMMDD_HHMMSS.log), so lexical sort
    is chronological.
    """
    logs = sorted(Path(log_dir).glob("alchemydetect_*.log"))
    for old in logs[: max(0, len(logs) - keep)]:
        try:
            old.unlink()
        except OSError:
            pass


def init_logging(level=logging.INFO):
    """Configure a session log file under the logs directory.

    Safe to call more than once: a second call does not add a duplicate session
    file handler.

    Returns:
        Path to the session log file.
    """
    log_dir = get_log_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"alchemydetect_{timestamp}.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if not any(getattr(h, _SESSION_MARKER, False) for h in logger.handlers):
        handler = logging.FileHandler(log_path, encoding="utf-8")
        setattr(handler, _SESSION_MARKER, True)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
        logger.info("AlchemyDetect session log started: %s", log_path)
    prune_old_logs(log_dir)
    return log_path


def get_logger():
    """Return the shared application logger."""
    return logging.getLogger(LOGGER_NAME)

"""Tests for application file logging."""

import logging

from alchemydetect.core.app_logging import (
    LOGGER_NAME,
    get_log_dir,
    get_logger,
    init_logging,
    prune_old_logs,
)

_SESSION_MARKER = "_alchemy_session"


def _reset_logger():
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _session_handlers():
    logger = logging.getLogger(LOGGER_NAME)
    return [h for h in logger.handlers if getattr(h, _SESSION_MARKER, False)]


def test_get_log_dir_env_override(tmp_path, monkeypatch):
    target = tmp_path / "mylogs"
    monkeypatch.setenv("ALCHEMYDETECT_LOG_DIR", str(target))
    log_dir = get_log_dir()
    assert log_dir == target
    assert log_dir.is_dir()


def test_init_logging_creates_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ALCHEMYDETECT_LOG_DIR", str(tmp_path))
    _reset_logger()
    try:
        log_path = init_logging()
        assert log_path.exists()
        assert log_path.parent == tmp_path
        # A message written through the shared logger lands in the file.
        get_logger().info("hello from test")
        for handler in _session_handlers():
            handler.flush()
        assert "hello from test" in log_path.read_text(encoding="utf-8")
    finally:
        _reset_logger()


def test_prune_old_logs_keeps_newest(tmp_path):
    # 25 timestamped log files; prune to keep the newest 20.
    names = [f"alchemydetect_202605{d:02d}_000000.log" for d in range(1, 26)]
    for n in names:
        (tmp_path / n).write_text("x", encoding="utf-8")
    prune_old_logs(tmp_path, keep=20)
    remaining = sorted(p.name for p in tmp_path.glob("alchemydetect_*.log"))
    assert len(remaining) == 20
    assert remaining == sorted(names)[-20:]  # the newest 20 survive


def test_init_logging_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("ALCHEMYDETECT_LOG_DIR", str(tmp_path))
    _reset_logger()
    try:
        init_logging()
        init_logging()
        assert len(_session_handlers()) == 1
    finally:
        _reset_logger()

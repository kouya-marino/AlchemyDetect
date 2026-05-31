"""Tests for the export worker process manager."""

from alchemydetect.workers.export_worker import ExportProcess


def test_export_process_initial_state():
    ep = ExportProcess()
    assert not ep.is_alive()
    assert ep.poll_metrics() == []


def test_export_process_cleanup_when_not_running():
    ep = ExportProcess()
    ep.cleanup()
    assert ep._process is None
    assert ep._message_queue is None
    assert ep._stop_event is None


def test_drain_remaining_no_process():
    ep = ExportProcess()
    assert ep.drain_remaining() == []

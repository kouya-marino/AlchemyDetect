"""Tests for the SpawnProcess base manager (lifecycle without spawning a child)."""

from alchemydetect.workers.spawn_process import SpawnProcess


def test_initial_state():
    sp = SpawnProcess()
    assert not sp.is_alive()
    assert sp.poll_metrics() == []
    assert sp.drain_remaining() == []


def test_request_stop_no_process_is_noop():
    SpawnProcess().request_stop()  # must not raise when there's no stop event yet


def test_terminate_when_not_running():
    sp = SpawnProcess()
    sp.terminate()
    assert sp._process is None
    assert sp._queue is None
    assert sp._stop_event is None


def test_cleanup_when_not_running():
    sp = SpawnProcess()
    sp.cleanup()
    assert sp._process is None
    assert sp._queue is None
    assert sp._stop_event is None

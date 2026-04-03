"""Tests for train worker module."""

from alchemydetect.workers.train_worker import TrainProcess


def test_train_process_initial_state():
    tp = TrainProcess()
    assert not tp.is_alive()
    assert tp.poll_metrics() == []


def test_train_process_cleanup_when_not_running():
    tp = TrainProcess()
    tp.cleanup()
    assert tp._process is None
    assert tp._metric_queue is None
    assert tp._stop_event is None

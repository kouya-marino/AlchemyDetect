"""Base manager for a `spawn` child process that streams dict messages via a Queue.

Shared by the training and export workers. Subclasses implement ``start(...)`` by
assembling the child entry function's payload and calling ``_spawn(target, payload)``;
the entry function must accept the message Queue and a stop Event as its final two
positional arguments.
"""

import multiprocessing as mp
from queue import Empty


class SpawnProcess:
    """Manages a spawn child process and a Queue/Event channel to it."""

    def __init__(self):
        self._process = None
        self._queue = None
        self._stop_event = None

    def _spawn(self, target, payload):
        """Start the child with args = (*payload, queue, stop_event)."""
        if self.is_alive():
            raise RuntimeError("A process is already running")
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue()
        self._stop_event = ctx.Event()
        self._process = ctx.Process(
            target=target,
            args=(*payload, self._queue, self._stop_event),
            daemon=False,
        )
        self._process.start()

    def request_stop(self):
        """Signal the child to stop gracefully (cooperative)."""
        if self._stop_event is not None:
            self._stop_event.set()

    def is_alive(self):
        """Check if the child process is still running."""
        return self._process is not None and self._process.is_alive()

    def poll_metrics(self):
        """Drain all currently-available messages from the queue.

        Returns:
            List of message dicts.
        """
        messages = []
        if self._queue is None:
            return messages
        while True:
            try:
                messages.append(self._queue.get_nowait())
            except Empty:
                break
        return messages

    def drain_remaining(self, timeout=2.0):
        """Join the finished process and drain any messages still in flight.

        A multiprocessing.Queue uses a background feeder thread, so terminal
        messages can still be in transit after the process is no longer alive.
        Joining flushes the feeder, after which a final drain returns the rest.
        """
        if self._process is not None:
            self._process.join(timeout=timeout)
        return self.poll_metrics()

    def terminate(self, timeout=3.0):
        """Forcefully stop and reap the child (used on app shutdown)."""
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=timeout)
        self._process = None
        self._queue = None
        self._stop_event = None

    def cleanup(self):
        """Reap a finished child and release resources."""
        if self._process is not None and not self._process.is_alive():
            self._process.join(timeout=5)
            self._process = None
        self._queue = None
        self._stop_event = None

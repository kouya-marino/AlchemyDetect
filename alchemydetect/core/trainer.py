"""Custom Detectron2 trainer with metric emission for the GUI."""

import logging
import sys

from detectron2.engine import DefaultTrainer, HookBase


class MetricEmitterHook(HookBase):
    """Hook that pushes training metrics to a multiprocessing.Queue."""

    def __init__(self, queue, stop_event, period=20):
        """
        Args:
            queue: multiprocessing.Queue to push metric dicts into.
            stop_event: multiprocessing.Event; when set, training stops.
            period: Emit metrics every N iterations.
        """
        self._queue = queue
        self._stop_event = stop_event
        self._period = period

    def after_step(self):
        # Check for stop request
        if self._stop_event.is_set():
            self._queue.put({"type": "log", "msg": "Training stopped by user."})
            sys.exit(0)

        iter_num = self.trainer.iter
        if (iter_num + 1) % self._period == 0:
            storage = self.trainer.storage
            metrics = {}
            for k, v in storage.latest().items():
                if isinstance(v, tuple):
                    metrics[k] = v[0]
            metrics["iter"] = iter_num + 1
            metrics["type"] = "metrics"
            self._queue.put(metrics)


class QueueLogHandler(logging.Handler):
    """Logging handler that sends log records to a multiprocessing.Queue."""

    def __init__(self, queue):
        super().__init__()
        self._queue = queue

    def emit(self, record):
        try:
            msg = self.format(record)
            self._queue.put({"type": "log", "msg": msg})
        except Exception:
            pass


class AlchemyTrainer(DefaultTrainer):
    """Detectron2 trainer that emits metrics to a queue for GUI consumption."""

    def __init__(self, cfg, metric_queue, stop_event):
        self._metric_queue = metric_queue
        self._stop_event = stop_event
        super().__init__(cfg)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(MetricEmitterHook(self._metric_queue, self._stop_event))
        return hooks

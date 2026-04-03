"""Background process for Detectron2 training.

Training runs in a separate process to avoid GIL issues and CUDA context conflicts.
Communication with the GUI happens via multiprocessing.Queue (metrics/logs)
and multiprocessing.Event (stop signal).
"""

import logging
import multiprocessing as mp
import os
import traceback
from queue import Empty


def _train_process_entry(cfg_yaml, output_dir, metric_queue, stop_event):
    """Entry point for the training child process.

    Args:
        cfg_yaml: Serialized Detectron2 config (YAML string).
        output_dir: Output directory path.
        metric_queue: multiprocessing.Queue for sending metrics/logs to parent.
        stop_event: multiprocessing.Event for receiving stop signal from parent.
    """
    try:
        # Import heavy deps only in the child process
        from detectron2.config import get_cfg

        from alchemydetect.core.trainer import AlchemyTrainer, QueueLogHandler

        # Set up logging to redirect to the queue
        queue_handler = QueueLogHandler(metric_queue)
        queue_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S"))

        root_logger = logging.getLogger()
        root_logger.addHandler(queue_handler)
        root_logger.setLevel(logging.INFO)

        # Also capture detectron2 logger
        d2_logger = logging.getLogger("detectron2")
        d2_logger.addHandler(queue_handler)
        d2_logger.setLevel(logging.INFO)

        # Reconstruct config
        cfg = get_cfg()
        cfg.merge_from_string(cfg_yaml)
        cfg.OUTPUT_DIR = output_dir
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        cfg.freeze()

        # Save config for later inference use
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            f.write(cfg.dump())

        metric_queue.put({"type": "log", "msg": "Training started..."})
        metric_queue.put({"type": "status", "status": "running"})

        trainer = AlchemyTrainer(cfg, metric_queue, stop_event)
        trainer.resume_or_load(resume=False)
        trainer.train()

        metric_queue.put({"type": "log", "msg": "Training completed successfully!"})
        metric_queue.put({"type": "status", "status": "completed"})

    except SystemExit:
        metric_queue.put({"type": "status", "status": "stopped"})
    except Exception:
        tb = traceback.format_exc()
        metric_queue.put({"type": "log", "msg": f"Training error:\n{tb}"})
        metric_queue.put({"type": "status", "status": "error"})


class TrainProcess:
    """Manages a child process that runs Detectron2 training."""

    def __init__(self):
        self._process = None
        self._metric_queue = None
        self._stop_event = None

    def start(self, cfg):
        """Start training in a child process.

        Args:
            cfg: A Detectron2 CfgNode (will be serialized to YAML).
        """
        if self.is_alive():
            raise RuntimeError("Training is already running")

        ctx = mp.get_context("spawn")
        self._metric_queue = ctx.Queue()
        self._stop_event = ctx.Event()

        cfg_yaml = cfg.dump()
        output_dir = cfg.OUTPUT_DIR

        self._process = ctx.Process(
            target=_train_process_entry,
            args=(cfg_yaml, output_dir, self._metric_queue, self._stop_event),
            daemon=True,
        )
        self._process.start()

    def request_stop(self):
        """Signal the child process to stop gracefully."""
        if self._stop_event is not None:
            self._stop_event.set()

    def is_alive(self):
        """Check if the training process is still running."""
        return self._process is not None and self._process.is_alive()

    def poll_metrics(self):
        """Drain all available messages from the metric queue.

        Returns:
            List of message dicts.
        """
        messages = []
        if self._metric_queue is None:
            return messages
        while True:
            try:
                msg = self._metric_queue.get_nowait()
                messages.append(msg)
            except Empty:
                break
        return messages

    def cleanup(self):
        """Clean up resources after training is done."""
        if self._process is not None and not self._process.is_alive():
            self._process.join(timeout=5)
            self._process = None
        self._metric_queue = None
        self._stop_event = None

"""Background process for exporting Detectron2 models to deployment formats.

Export runs in a separate process (mirroring train_worker) because it imports
torch / detectron2 and may touch CUDA; process isolation keeps the GUI responsive
and ensures a tracing crash never takes down the application. Communication with
the GUI uses multiprocessing.Queue (logs / artifacts / status) and a
multiprocessing.Event (stop request).
"""

import multiprocessing as mp
import traceback
from queue import Empty


def _export_process_entry(resolved, output_dir, fmt, options, message_queue, stop_event):
    """Entry point for the export child process.

    Args:
        resolved: Dict from exporter.resolve_model_dir (model_dir, weights_path, ...).
        output_dir: Destination directory for export artifacts.
        fmt: Export format ("onnx"; "tensorrt" added in a later phase).
        options: Dict of format-specific options (opset, input_size, fp16, ...).
        message_queue: multiprocessing.Queue for sending messages to the parent.
        stop_event: multiprocessing.Event signalling a stop request.
    """
    try:
        # Import heavy deps only in the child process.
        import torch

        from alchemydetect.core.exporter import run_onnx_export

        def log(msg):
            message_queue.put({"type": "log", "msg": msg})

        # Decide and report the compute device in the child, never in the GUI.
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            device_str = f"GPU: {gpu_name} ({gpu_mem:.1f} GB)"
        else:
            device_str = "CPU (no GPU available)"
        message_queue.put({"type": "device", "device": device_str})
        log(f"Device: {device_str}")

        if stop_event.is_set():
            message_queue.put({"type": "status", "status": "stopped"})
            return

        message_queue.put({"type": "status", "status": "running"})

        if fmt == "onnx":
            artifacts = run_onnx_export(resolved, output_dir, options, log)
        else:
            raise ValueError(f"Unsupported export format: {fmt}")

        for path in artifacts:
            message_queue.put({"type": "artifact", "path": path})

        log("Export completed successfully!")
        message_queue.put({"type": "status", "status": "completed"})

    except Exception:
        tb = traceback.format_exc()
        message_queue.put({"type": "log", "msg": f"Export error:\n{tb}"})
        message_queue.put({"type": "status", "status": "error"})


class ExportProcess:
    """Manages a child process that exports a model to a deployment format."""

    def __init__(self):
        self._process = None
        self._message_queue = None
        self._stop_event = None

    def start(self, resolved, output_dir, fmt, options):
        """Start an export in a child process.

        Args:
            resolved: Dict from exporter.resolve_model_dir.
            output_dir: Destination directory for artifacts.
            fmt: Export format string.
            options: Dict of format-specific options.
        """
        if self.is_alive():
            raise RuntimeError("An export is already running")

        ctx = mp.get_context("spawn")
        self._message_queue = ctx.Queue()
        self._stop_event = ctx.Event()

        self._process = ctx.Process(
            target=_export_process_entry,
            args=(resolved, output_dir, fmt, options, self._message_queue, self._stop_event),
            daemon=False,
        )
        self._process.start()

    def request_stop(self):
        """Signal the child process to stop. Export only checks this at stage
        boundaries — the single torch.onnx.export call cannot be interrupted."""
        if self._stop_event is not None:
            self._stop_event.set()

    def is_alive(self):
        """Check if the export process is still running."""
        return self._process is not None and self._process.is_alive()

    def poll_metrics(self):
        """Drain all available messages from the message queue.

        Returns:
            List of message dicts.
        """
        messages = []
        if self._message_queue is None:
            return messages
        while True:
            try:
                messages.append(self._message_queue.get_nowait())
            except Empty:
                break
        return messages

    def drain_remaining(self, timeout=2.0):
        """Join the finished process and drain any messages still in flight.

        A multiprocessing.Queue uses a background feeder thread, so terminal
        messages can still be in transit after the process is no longer alive.
        Joining flushes the feeder, after which a final drain returns the rest.

        Returns:
            List of remaining message dicts.
        """
        if self._process is not None:
            self._process.join(timeout=timeout)
        return self.poll_metrics()

    def cleanup(self):
        """Clean up resources after export is done."""
        if self._process is not None and not self._process.is_alive():
            self._process.join(timeout=5)
            self._process = None
        self._message_queue = None
        self._stop_event = None

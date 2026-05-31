"""Background process for exporting Detectron2 models to deployment formats.

Export runs in a separate process (mirroring train_worker) because it imports
torch / detectron2 and may touch CUDA; process isolation keeps the GUI responsive
and ensures a tracing crash never takes down the application. Communication with
the GUI uses multiprocessing.Queue (logs / artifacts / status) and a
multiprocessing.Event (stop request).
"""

import traceback

from alchemydetect.workers.spawn_process import SpawnProcess


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

        from alchemydetect.core.exporter import run_onnx_export, run_tensorrt_export

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
        elif fmt == "tensorrt":
            artifacts = run_tensorrt_export(resolved, output_dir, options, log)
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


class ExportProcess(SpawnProcess):
    """Manages a child process that exports a model to a deployment format.

    Note: request_stop only takes effect at stage boundaries — the single
    torch.onnx.export / TensorRT build call cannot be interrupted mid-flight.
    """

    def start(self, resolved, output_dir, fmt, options):
        """Start an export in a child process.

        Args:
            resolved: Dict from exporter.resolve_model_dir.
            output_dir: Destination directory for artifacts.
            fmt: Export format string.
            options: Dict of format-specific options.
        """
        self._spawn(_export_process_entry, (resolved, output_dir, fmt, options))

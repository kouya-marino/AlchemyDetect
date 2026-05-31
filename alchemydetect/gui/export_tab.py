"""Export tab: load a trained model and export it to a deployment format (ONNX)."""

from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from alchemydetect.core.exporter import (
    detect_task_from_config,
    is_onnx_available,
    is_onnxruntime_available,
    is_tensorrt_available,
    read_class_names,
    resolve_model_dir,
)
from alchemydetect.workers.export_worker import ExportProcess

from .dialogs import browse_directory, load_model_dialog
from .log_viewer import LogViewer


class ExportTab(QWidget):
    """Tab for exporting a trained model to ONNX for faster deployment."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._resolved = None  # dict from resolve_model_dir
        self._export_process = ExportProcess()
        self._artifacts = []
        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Model group ---
        model_group = QGroupBox("Model")
        mg_layout = QHBoxLayout(model_group)
        self._model_label = QLabel("No model loaded")
        mg_layout.addWidget(self._model_label, stretch=1)
        load_btn = QPushButton("Load Model...")
        load_btn.clicked.connect(self._on_load_model)
        mg_layout.addWidget(load_btn)
        main_layout.addWidget(model_group)

        # Model info / experimental warning label
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: #2e7d32; font-weight: bold; padding: 4px;")
        main_layout.addWidget(self._info_label)

        # --- Format + options row ---
        config_row = QHBoxLayout()

        format_group = QGroupBox("Format")
        fg_layout = QVBoxLayout(format_group)
        self._format_combo = QComboBox()
        formats = ["ONNX"]
        if is_tensorrt_available():
            formats.append("TensorRT")
        self._format_combo.addItems(formats)
        self._format_combo.currentTextChanged.connect(self._on_format_changed)
        fg_layout.addWidget(self._format_combo)
        # TensorRT requires building an ONNX first, then the engine.
        self._trt_workspace_spin = QDoubleSpinBox()
        self._trt_workspace_spin.setRange(0.25, 64.0)
        self._trt_workspace_spin.setSingleStep(0.5)
        self._trt_workspace_spin.setValue(4.0)
        self._trt_workspace_spin.setSuffix(" GB workspace")
        self._trt_workspace_spin.setVisible(False)
        fg_layout.addWidget(self._trt_workspace_spin)
        config_row.addWidget(format_group)

        opt_group = QGroupBox("ONNX Options")
        opt_layout = QHBoxLayout(opt_group)

        opt_layout.addWidget(QLabel("Opset:"))
        self._opset_spin = QSpinBox()
        self._opset_spin.setRange(11, 19)
        self._opset_spin.setValue(17)
        opt_layout.addWidget(self._opset_spin)

        opt_layout.addWidget(QLabel("Input H:"))
        self._height_spin = QSpinBox()
        self._height_spin.setRange(128, 2048)
        self._height_spin.setSingleStep(32)
        self._height_spin.setValue(800)
        opt_layout.addWidget(self._height_spin)

        opt_layout.addWidget(QLabel("W:"))
        self._width_spin = QSpinBox()
        self._width_spin.setRange(128, 2048)
        self._width_spin.setSingleStep(32)
        self._width_spin.setValue(800)
        opt_layout.addWidget(self._width_spin)

        self._dynamic_check = QCheckBox("Dynamic axes")
        self._dynamic_check.setChecked(True)
        opt_layout.addWidget(self._dynamic_check)

        self._fp16_check = QCheckBox("fp16")
        opt_layout.addWidget(self._fp16_check)

        self._validate_check = QCheckBox("Validate (onnxruntime)")
        if is_onnxruntime_available():
            self._validate_check.setChecked(True)
        else:
            self._validate_check.setEnabled(False)
            self._validate_check.setToolTip(
                "Install onnxruntime to enable validation (pip install alchemydetect[export])"
            )
        opt_layout.addWidget(self._validate_check)

        config_row.addWidget(opt_group, stretch=1)
        main_layout.addLayout(config_row)

        # --- Output directory ---
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output Dir:"))
        self._output_dir_edit = QLineEdit()
        self._output_dir_edit.setPlaceholderText("Directory to write the exported model")
        out_row.addWidget(self._output_dir_edit)
        btn_output = QPushButton("Browse...")
        btn_output.clicked.connect(lambda: self._browse_output())
        out_row.addWidget(btn_output)
        main_layout.addLayout(out_row)

        # --- Control buttons ---
        btn_row = QHBoxLayout()
        self._export_btn = QPushButton("Export")
        self._export_btn.clicked.connect(self._on_export)
        btn_row.addWidget(self._export_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel_btn)

        self._device_label = QLabel("")
        self._device_label.setStyleSheet("font-weight: bold; padding: 2px 6px;")
        btn_row.addWidget(self._device_label)
        btn_row.addStretch(1)
        main_layout.addLayout(btn_row)

        # --- Progress + log ---
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        main_layout.addWidget(self._progress)

        self._log_viewer = LogViewer()
        main_layout.addWidget(self._log_viewer, stretch=1)

    def _setup_timer(self):
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(150)
        self._poll_timer.timeout.connect(self._poll_export)

    def _on_format_changed(self, text):
        self._trt_workspace_spin.setVisible(text == "TensorRT")

    def _browse_output(self):
        path = browse_directory(self, start_dir=self._output_dir_edit.text())
        if path:
            self._output_dir_edit.setText(path)

    def _on_load_model(self):
        config_path, weights_path = load_model_dialog(self)
        if not (config_path and weights_path):
            return
        try:
            self._resolved = resolve_model_dir(weights_path)
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Load Model", str(e))
            self._resolved = None
            return

        model_dir = self._resolved["model_dir"]
        name = Path(model_dir).name
        self._model_label.setText(f"Loaded: {name}/{Path(weights_path).name}")

        class_names = read_class_names(model_dir)
        task = detect_task_from_config(self._resolved["config_path"])
        parts = [f"{len(class_names)} classes" if class_names else "class names not found"]
        if task == "instance_segmentation":
            parts.append("instance segmentation — ONNX export is EXPERIMENTAL")
            self._info_label.setStyleSheet("color: #d84315; font-weight: bold; padding: 4px;")
        else:
            self._info_label.setStyleSheet("color: #2e7d32; font-weight: bold; padding: 4px;")
        self._info_label.setText(" | ".join(parts))

    def _on_export(self):
        if not self._resolved:
            QMessageBox.warning(self, "No Model", "Please load a model first.")
            return
        output_dir = self._output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Missing Output", "Please choose an output directory.")
            return

        fmt = "tensorrt" if self._format_combo.currentText() == "TensorRT" else "onnx"

        if not is_onnx_available():
            QMessageBox.critical(
                self,
                "ONNX Not Installed",
                "Export requires the 'onnx' package, which is not installed.\n\n"
                "Install the export extras with:\n    pip install alchemydetect[export]",
            )
            return
        if fmt == "tensorrt" and not is_tensorrt_available():
            QMessageBox.critical(
                self,
                "TensorRT Not Installed",
                "TensorRT export requires the 'tensorrt' package, which is not installed.\n\n"
                "TensorRT is not pip-installable from this project — install it manually to "
                "match your CUDA/cuDNN versions (see INSTALL.md).",
            )
            return

        options = {
            "opset": self._opset_spin.value(),
            "input_size": (self._height_spin.value(), self._width_spin.value()),
            "fp16": self._fp16_check.isChecked(),
            "dynamic_axes": self._dynamic_check.isChecked(),
            "validate": self._validate_check.isChecked(),
            "workspace_gb": self._trt_workspace_spin.value(),
        }

        self._artifacts = []
        self._log_viewer.clear_logs()
        self._log_viewer.append_log(f"--- Starting {fmt.upper()} export ---")

        try:
            self._export_process.start(self._resolved, output_dir, fmt, options)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
            return

        self._export_btn.setEnabled(False)
        self._export_btn.setText("Exporting...")
        self._cancel_btn.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # busy indicator
        self._poll_timer.start()

    def _on_cancel(self):
        self._export_process.request_stop()
        self._cancel_btn.setEnabled(False)
        self._log_viewer.append_log("Cancel requested (takes effect at the next stage boundary)...")

    def _poll_export(self):
        if self._handle_messages(self._export_process.poll_metrics()):
            return
        # Process died: drain any in-flight terminal message before concluding.
        if not self._export_process.is_alive() and self._poll_timer.isActive():
            if self._handle_messages(self._export_process.drain_remaining()):
                return
            self._on_export_finished("error")

    def _handle_messages(self, messages):
        """Process a batch of messages. Returns True if a terminal status was handled."""
        for msg in messages:
            msg_type = msg.get("type", "")
            if msg_type == "log":
                self._log_viewer.append_log(msg.get("msg", ""))
            elif msg_type == "device":
                device_str = msg.get("device", "")
                color = "#2e7d32" if "GPU" in device_str else "#d84315"
                self._device_label.setStyleSheet(f"color: {color}; font-weight: bold; padding: 2px 6px;")
                self._device_label.setText(device_str)
            elif msg_type == "artifact":
                self._artifacts.append(msg.get("path", ""))
            elif msg_type == "status":
                status = msg.get("status", "")
                if status == "running":
                    self._export_btn.setText("Exporting...")
                elif status in ("completed", "stopped", "error"):
                    self._on_export_finished(status)
                    return True
        return False

    def _on_export_finished(self, status):
        self._poll_timer.stop()
        self._export_btn.setText("Export")
        self._export_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setVisible(False)
        self._progress.setRange(0, 1)
        self._export_process.cleanup()

        if status == "completed":
            self._log_viewer.append_log("--- Export completed! ---")
            files = "\n".join(self._artifacts) if self._artifacts else "(no files reported)"
            QMessageBox.information(self, "Export Complete", f"Exported files:\n{files}")
        elif status == "stopped":
            self._log_viewer.append_log("--- Export cancelled ---")
        else:
            self._log_viewer.append_log("--- Export ended with errors ---")

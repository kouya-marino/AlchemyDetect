"""Training tab: dataset selection, model config, training controls, live metrics."""

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from alchemydetect.core.model_catalog import get_model_names
from alchemydetect.workers.train_worker import TrainProcess

from .dialogs import browse_directory, browse_file, save_model_dialog
from .log_viewer import LogViewer
from .loss_plot import LossPlot


class TrainTab(QWidget):
    """Tab for configuring and running Detectron2 training."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._train_process = TrainProcess()
        self._last_output_dir = ""
        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Dataset group ---
        dataset_group = QGroupBox("Dataset (COCO Format)")
        dg_layout = QVBoxLayout(dataset_group)

        # Train images
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Train Images:"))
        self._train_images_edit = QLineEdit()
        self._train_images_edit.setPlaceholderText("Path to training images directory")
        row1.addWidget(self._train_images_edit)
        btn_train_img = QPushButton("Browse...")
        btn_train_img.clicked.connect(lambda: self._browse_to(self._train_images_edit, directory=True))
        row1.addWidget(btn_train_img)
        dg_layout.addLayout(row1)

        # Train annotations
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Train JSON:"))
        self._train_json_edit = QLineEdit()
        self._train_json_edit.setPlaceholderText("Path to COCO annotations JSON")
        row2.addWidget(self._train_json_edit)
        btn_train_json = QPushButton("Browse...")
        btn_train_json.clicked.connect(lambda: self._browse_to(self._train_json_edit, directory=False))
        row2.addWidget(btn_train_json)
        dg_layout.addLayout(row2)

        # Validation (optional)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Val Images:"))
        self._val_images_edit = QLineEdit()
        self._val_images_edit.setPlaceholderText("(Optional) Validation images directory")
        row3.addWidget(self._val_images_edit)
        btn_val_img = QPushButton("Browse...")
        btn_val_img.clicked.connect(lambda: self._browse_to(self._val_images_edit, directory=True))
        row3.addWidget(btn_val_img)
        dg_layout.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Val JSON:"))
        self._val_json_edit = QLineEdit()
        self._val_json_edit.setPlaceholderText("(Optional) Validation annotations JSON")
        row4.addWidget(self._val_json_edit)
        btn_val_json = QPushButton("Browse...")
        btn_val_json.clicked.connect(lambda: self._browse_to(self._val_json_edit, directory=False))
        row4.addWidget(btn_val_json)
        dg_layout.addLayout(row4)

        # Dataset info label
        self._dataset_info_label = QLabel("")
        self._dataset_info_label.setStyleSheet("color: #2e7d32; font-weight: bold; padding: 4px;")
        dg_layout.addWidget(self._dataset_info_label)

        # Update info when JSON path changes
        self._train_json_edit.textChanged.connect(self._update_dataset_info)

        main_layout.addWidget(dataset_group)

        # --- Model + Hyperparameters row ---
        config_row = QHBoxLayout()

        # Model group
        model_group = QGroupBox("Model")
        mg_layout = QVBoxLayout(model_group)
        self._model_combo = QComboBox()
        self._model_combo.addItems(get_model_names())
        mg_layout.addWidget(self._model_combo)
        config_row.addWidget(model_group)

        # Hyperparameters group
        hp_group = QGroupBox("Hyperparameters")
        hp_layout = QHBoxLayout(hp_group)

        hp_layout.addWidget(QLabel("LR:"))
        self._lr_spin = QDoubleSpinBox()
        self._lr_spin.setDecimals(6)
        self._lr_spin.setRange(0.000001, 1.0)
        self._lr_spin.setValue(0.0025)
        self._lr_spin.setSingleStep(0.0005)
        hp_layout.addWidget(self._lr_spin)

        hp_layout.addWidget(QLabel("Iterations:"))
        self._iter_spin = QSpinBox()
        self._iter_spin.setRange(10, 1000000)
        self._iter_spin.setValue(1000)
        self._iter_spin.setSingleStep(100)
        hp_layout.addWidget(self._iter_spin)

        hp_layout.addWidget(QLabel("Batch Size:"))
        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 64)
        self._batch_spin.setValue(2)
        hp_layout.addWidget(self._batch_spin)

        config_row.addWidget(hp_group)
        main_layout.addLayout(config_row)

        # --- Output directory ---
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output Dir:"))
        self._output_dir_edit = QLineEdit()
        self._output_dir_edit.setPlaceholderText("Directory to save training outputs")
        out_row.addWidget(self._output_dir_edit)
        btn_output = QPushButton("Browse...")
        btn_output.clicked.connect(lambda: self._browse_to(self._output_dir_edit, directory=True))
        out_row.addWidget(btn_output)
        main_layout.addLayout(out_row)

        # --- Control buttons ---
        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("Start Training")
        self._start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stop Training")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self._stop_btn)

        self._save_btn = QPushButton("Save Model")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(self._save_btn)

        # Device label
        self._device_label = QLabel("")
        self._device_label.setStyleSheet("font-weight: bold; padding: 2px 6px;")
        btn_row.addWidget(self._device_label)

        main_layout.addLayout(btn_row)

        # --- Log viewer + Loss plot (split view) ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._log_viewer = LogViewer()
        self._loss_plot = LossPlot()
        splitter.addWidget(self._log_viewer)
        splitter.addWidget(self._loss_plot)
        splitter.setSizes([500, 500])
        main_layout.addWidget(splitter, stretch=1)

    def _setup_timer(self):
        """Timer to poll training process for metrics/logs."""
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(150)
        self._poll_timer.timeout.connect(self._poll_training)

    def _browse_to(self, line_edit, directory=True):
        if directory:
            path = browse_directory(self, start_dir=line_edit.text())
        else:
            path = browse_file(self, start_dir=line_edit.text(), filter_str="JSON Files (*.json);;All Files (*)")
        if path:
            line_edit.setText(path)

    def _update_dataset_info(self):
        """Update the dataset info label when the JSON path changes."""
        json_path = self._train_json_edit.text().strip()
        if not json_path:
            self._dataset_info_label.setText("")
            return
        try:
            from alchemydetect.core.dataset_utils import get_dataset_summary

            summary = get_dataset_summary(json_path)
            classes_str = ", ".join(summary["class_names"])
            self._dataset_info_label.setText(
                f"{summary['num_images']} images | {summary['num_annotations']} annotations | "
                f"{summary['num_classes']} classes: {classes_str}"
            )
        except Exception:
            self._dataset_info_label.setText("")

    def _on_start(self):
        """Validate inputs and start training."""
        train_images = self._train_images_edit.text().strip()
        train_json = self._train_json_edit.text().strip()
        output_dir = self._output_dir_edit.text().strip()

        if not train_images or not train_json or not output_dir:
            QMessageBox.warning(self, "Missing Fields", "Please fill in Train Images, Train JSON, and Output Dir.")
            return

        # Validate dataset
        from alchemydetect.core.dataset_utils import get_dataset_summary, validate_coco_json

        is_valid, msg = validate_coco_json(train_json, train_images)
        if not is_valid:
            QMessageBox.critical(self, "Dataset Error", msg)
            return

        val_images = self._val_images_edit.text().strip() or None
        val_json = self._val_json_edit.text().strip() or None

        model_name = self._model_combo.currentText()

        # Show dataset summary in log
        summary = get_dataset_summary(train_json)
        classes_str = ", ".join(summary["class_names"])
        self._log_viewer.append_log("--- Dataset Summary ---")
        self._log_viewer.append_log(f"  Images: {summary['num_images']}")
        self._log_viewer.append_log(f"  Annotations: {summary['num_annotations']}")
        self._log_viewer.append_log(f"  Classes ({summary['num_classes']}): {classes_str}")
        self._log_viewer.append_log(f"  Model: {model_name}")
        self._log_viewer.append_log(f"  LR: {self._lr_spin.value()} | Iterations: {self._iter_spin.value()} | Batch: {self._batch_spin.value()}")
        self._log_viewer.append_log("")

        try:
            from alchemydetect.core.config_builder import build_cfg

            cfg = build_cfg(
                model_name=model_name,
                train_images_dir=train_images,
                train_json=train_json,
                output_dir=output_dir,
                lr=self._lr_spin.value(),
                max_iter=self._iter_spin.value(),
                batch_size=self._batch_spin.value(),
                val_images_dir=val_images,
                val_json=val_json,
            )
        except Exception as e:
            QMessageBox.critical(self, "Config Error", str(e))
            return

        self._last_output_dir = output_dir
        self._log_viewer.clear_logs()
        self._loss_plot.clear_plot()

        # Build dataset info for the child process to re-register
        dataset_info = [
            {"name": "alchemy_train", "json_path": train_json, "image_root": train_images},
        ]
        if val_images and val_json:
            dataset_info.append(
                {"name": "alchemy_val", "json_path": val_json, "image_root": val_images},
            )

        try:
            self._train_process.start(cfg, dataset_info)
        except Exception as e:
            QMessageBox.critical(self, "Training Error", str(e))
            return

        self._start_btn.setEnabled(False)
        self._start_btn.setText("Preparing...")
        self._stop_btn.setEnabled(True)
        self._save_btn.setEnabled(False)
        self._poll_timer.start()

    def _on_stop(self):
        """Request training to stop."""
        self._train_process.request_stop()
        self._stop_btn.setEnabled(False)
        self._log_viewer.append_log("Stop requested... waiting for training to finish current iteration.")

    def _on_save(self):
        """Save the trained model."""
        if self._last_output_dir:
            save_model_dialog(self, self._last_output_dir)

    def _poll_training(self):
        """Poll the training process for new metrics and logs."""
        messages = self._train_process.poll_metrics()

        for msg in messages:
            msg_type = msg.get("type", "")
            if msg_type == "log":
                self._log_viewer.append_log(msg.get("msg", ""))
            elif msg_type == "device":
                device_str = msg.get("device", "")
                if "GPU" in device_str:
                    self._device_label.setStyleSheet("color: #2e7d32; font-weight: bold; padding: 2px 6px;")
                else:
                    self._device_label.setStyleSheet("color: #d84315; font-weight: bold; padding: 2px 6px;")
                self._device_label.setText(device_str)
            elif msg_type == "metrics":
                total_loss = msg.get("total_loss")
                iteration = msg.get("iter")
                if total_loss is not None and iteration is not None:
                    self._loss_plot.add_point(iteration, total_loss)
                    self._log_viewer.append_log(f"[Iter {iteration}] total_loss={total_loss:.4f}")
            elif msg_type == "status":
                status = msg.get("status", "")
                if status == "downloading":
                    self._start_btn.setText("Downloading weights...")
                elif status == "running":
                    self._start_btn.setText("Training...")
                elif status in ("completed", "stopped", "error"):
                    self._on_training_finished(status)

        # Check if process died unexpectedly
        if not self._train_process.is_alive() and self._poll_timer.isActive():
            self._on_training_finished("error")

    def _on_training_finished(self, status):
        """Handle training completion."""
        self._poll_timer.stop()
        self._start_btn.setText("Start Training")
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._save_btn.setEnabled(status == "completed")
        self._train_process.cleanup()

        if status == "completed":
            self._log_viewer.append_log("--- Training completed! ---")
        elif status == "stopped":
            self._log_viewer.append_log("--- Training stopped by user ---")
        else:
            self._log_viewer.append_log("--- Training ended with errors ---")

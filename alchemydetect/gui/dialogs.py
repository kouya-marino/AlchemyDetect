"""File dialogs for model save/load and dataset selection."""

import shutil
from pathlib import Path

from PyQt6.QtWidgets import QFileDialog, QMessageBox


def browse_directory(parent, title="Select Directory", start_dir=""):
    """Open a directory picker dialog. Returns path string or empty string."""
    path = QFileDialog.getExistingDirectory(parent, title, start_dir)
    return path


def browse_file(parent, title="Select File", start_dir="", filter_str="All Files (*)"):
    """Open a file picker dialog. Returns path string or empty string."""
    path, _ = QFileDialog.getOpenFileName(parent, title, start_dir, filter_str)
    return path


def save_model_dialog(parent, output_dir):
    """Save trained model (.pth + config.yaml) to a user-chosen directory.

    Args:
        parent: Parent widget.
        output_dir: The training output directory containing model_final.pth and config.yaml.

    Returns:
        Path to the saved directory, or None if cancelled.
    """
    output_path = Path(output_dir)
    weights_file = output_path / "model_final.pth"
    config_file = output_path / "config.yaml"

    if not weights_file.exists():
        QMessageBox.warning(parent, "Save Model", "No model_final.pth found. Train a model first.")
        return None

    dest_dir = QFileDialog.getExistingDirectory(parent, "Save Model To Directory")
    if not dest_dir:
        return None

    dest = Path(dest_dir)
    class_names_file = output_path / "class_names.json"
    try:
        shutil.copy2(weights_file, dest / "model_final.pth")
        if config_file.exists():
            shutil.copy2(config_file, dest / "config.yaml")
        if class_names_file.exists():
            shutil.copy2(class_names_file, dest / "class_names.json")
        QMessageBox.information(parent, "Save Model", f"Model saved to:\n{dest}")
        return str(dest)
    except Exception as e:
        QMessageBox.critical(parent, "Save Model Error", str(e))
        return None


def load_model_dialog(parent):
    """Open dialogs to select a model weights file and its config.

    Returns:
        (config_yaml_path, weights_path) or (None, None) if cancelled.
    """
    weights_path, _ = QFileDialog.getOpenFileName(
        parent, "Select Model Weights", "", "PyTorch Weights (*.pth);;All Files (*)"
    )
    if not weights_path:
        return None, None

    # Try to auto-find config.yaml in the same directory
    weights_dir = Path(weights_path).parent
    auto_config = weights_dir / "config.yaml"

    if auto_config.exists():
        return str(auto_config), weights_path

    # Ask user to select config manually
    config_path, _ = QFileDialog.getOpenFileName(
        parent, "Select Config YAML", str(weights_dir), "YAML Files (*.yaml *.yml);;All Files (*)"
    )
    if not config_path:
        return None, None

    return config_path, weights_path

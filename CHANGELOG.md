# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- The Inference and Deploy tabs now show the per-image detection time (and FPS)
  in the results side panel.
- The Deploy tab shows the active runtime provider (e.g. `onnxruntime —
  CPUExecutionProvider` / `CUDAExecutionProvider`, or `TensorRT`) so it's clear
  whether inference is running on CPU or GPU.
- The Export tab exposes the model's baked score threshold (default 0.05).
  Lower keeps the Deploy threshold slider flexible; set it to 0.5 to match the
  Detectron2 predictor's default for apples-to-apples comparisons.

### Fixed
- ONNX export mislabeled the traced `image_size` output (shape `(2,)`) as
  "classes" and gave it a duplicate name, so the Deploy tab crashed with
  "boolean index did not match indexed array" (the image-size tensor overwrote
  the real predicted classes). Output roles are now assigned by matching the
  detection count, with unique names and the image-size tensor ignored. Models
  exported before this fix should be re-exported; the Deploy runtime now
  substitutes defaults and logs a warning instead of crashing on them.

## [0.4.0] - 2026-05-31

### Added
- TensorRT export: when the `tensorrt` package is installed, the Export tab
  offers a TensorRT format that builds the ONNX first, then a `model.engine`
  (FP16 toggle + workspace-size option, dynamic-shape optimization profile).
  The Deploy tab can load and run `.engine` files via a TensorRT runtime
  (requires `tensorrt` + `pycuda`; experimental). TensorRT is gated behind a
  local install and never imported unless present.
- Deploy tab to run inference with exported ONNX models via `onnxruntime`
  (GPU provider used when available), independent of Detectron2's predictor.
  Loads `model.onnx` + its `export_metadata.json`, reproduces test-time resize,
  scales boxes back to the original image, and reuses the existing visualizer /
  detections table. Mask overlay is best-effort (experimental).
- Export tab to convert trained models to ONNX for faster deployment. Writes
  `model.onnx` plus the copied `config.yaml` / `class_names.json` and an
  `export_metadata.json` (records opset, input size, preprocessing params, and
  output tensor roles). Detection models export reliably; Mask R-CNN export is
  experimental. Install support with `pip install alchemydetect[export]`.
- `device` override parameter on `load_predictor` so model loading outside the
  GUI process (e.g. export) can target CPU/GPU based on local availability.
- Persistent session logging to a `logs/` directory (override with
  `ALCHEMYDETECT_LOG_DIR`). Training/export output and inference errors —
  including relayed worker tracebacks — are mirrored to a timestamped log file
  for post-hoc analysis. The active log path is shown in the status bar.

### Fixed
- ONNX export now fails fast with a clear "install alchemydetect[export]"
  message when the `onnx` package is missing, instead of a cryptic torch
  `OnnxExporterError` deep in the export call.
- Deploy tab now falls back to CPU cleanly when the onnxruntime CUDA provider
  can't load (e.g. `onnxruntime-gpu` installed without the CUDA 12 / cuDNN 9
  runtime), silencing the noisy provider error and logging the active provider.
- Class names now ordered by COCO category id to match Detectron2's contiguous
  class id mapping — previously predictions could be mislabeled when categories
  were not listed in ascending-id order
- Successful short training runs no longer occasionally report "error": the
  training process is now joined and its queue drained for in-flight terminal
  status messages before concluding
- Compute device (CUDA/CPU) is now selected in the training child process
  instead of the GUI process, so building a config no longer initializes a
  CUDA context in the GUI process

## [0.3.0] - 2026-04-05

### Added
- Dataset info display in Train tab (images, annotations, class count and names)
- GPU/CPU device indicator label in training UI
- Class name mapping saved during training (`class_names.json`)
- Inference shows class names instead of IDs in overlay and detections table

### Fixed
- Training process crash due to daemon subprocess restriction (set `daemon=False`)
- Dataloader `NUM_WORKERS` set to 0 to avoid nested multiprocessing on Windows
- Dataset registration in child process (fixes `alchemy_train` not found error)
- Config reconstruction using `merge_from_file` instead of missing `merge_from_string`
- GPU memory attribute name (`total_memory` not `total_mem`)

## [0.2.0] - 2026-04-04

### Added
- CI/CD pipeline with GitHub Actions (lint, test, build)
- Automated PyPI publishing on version tags
- Test suite with 23 tests covering core modules and GUI widgets
- pyproject.toml for modern Python packaging
- ruff for linting and formatting
- Pretrained weights download status indicator in training UI

## [0.1.0] - 2026-04-04

### Added
- Initial project structure with core, workers, and GUI modules
- Training tab with dataset selection, model picker, hyperparameter controls, live logs, and loss plot
- Inference tab with model loading, single image/folder inference, and result navigation
- Support for Faster R-CNN, RetinaNet (Object Detection) and Mask R-CNN (Instance Segmentation)
- COCO JSON dataset format support with validation
- Background training via multiprocessing to keep GUI responsive
- Model save/load functionality (.pth + config.yaml pair)
- Real-time loss plotting with pyqtgraph

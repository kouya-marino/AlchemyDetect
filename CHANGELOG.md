# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Export tab to convert trained models to ONNX for faster deployment. Writes
  `model.onnx` plus the copied `config.yaml` / `class_names.json` and an
  `export_metadata.json` (records opset, input size, preprocessing params, and
  output tensor roles). Detection models export reliably; Mask R-CNN export is
  experimental. Install support with `pip install alchemydetect[export]`.
- `device` override parameter on `load_predictor` so model loading outside the
  GUI process (e.g. export) can target CPU/GPU based on local availability.

### Fixed
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

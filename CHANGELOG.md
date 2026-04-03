# Changelog

All notable changes to this project will be documented in this file.

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

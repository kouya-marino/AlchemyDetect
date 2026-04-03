# AlchemyDetect — Project Plan

## Overview
PyQt6 desktop application wrapping Detectron2 for training object detection / instance segmentation models and running inference on images.

## Features
- **Train Tab**: Select COCO dataset, pick model architecture, configure hyperparameters, start/stop training with live logs and loss plot
- **Inference Tab**: Load trained model, run inference on single image or folder, visualize results with bounding boxes/masks
- **Model Management**: Save trained weights + config, load for later inference

## Tech Stack
- Python 3.10/3.11
- PyQt6 (GUI)
- Detectron2 + PyTorch (ML)
- pyqtgraph (real-time loss plotting)
- COCO JSON (dataset format)

## Architecture

```
AlchemyDetect/
├── main.py                          # Entry point
├── requirements.txt
├── alchemydetect/
│   ├── core/                        # Non-GUI logic
│   │   ├── model_catalog.py         # Model zoo mappings
│   │   ├── config_builder.py        # Builds Detectron2 config from user params
│   │   ├── dataset_utils.py         # COCO dataset registration & validation
│   │   ├── trainer.py               # Custom trainer with metric hooks
│   │   └── inferencer.py            # Inference wrapper
│   ├── workers/                     # Background threads/processes
│   │   ├── train_worker.py          # Separate process for training
│   │   └── inference_worker.py      # QThread for inference
│   └── gui/                         # PyQt6 widgets
│       ├── main_window.py           # Main window with tabs
│       ├── train_tab.py             # Training UI
│       ├── inference_tab.py         # Inference UI
│       ├── log_viewer.py            # Live log display
│       ├── loss_plot.py             # Real-time loss chart
│       ├── image_viewer.py          # Image display with overlays
│       └── dialogs.py               # File dialogs for save/load
```

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Training isolation | `multiprocessing.Process` | Avoids GIL and CUDA context issues |
| IPC | `Queue` + `Event` | Simple, picklable, cross-platform |
| Config serialization | YAML via `cfg.dump()` | Avoids pickling issues |
| Loss plot | pyqtgraph | Faster real-time rendering than matplotlib |
| Model save format | `.pth` + `.yaml` pair | Standard Detectron2 convention |

## Implementation Status

### Phase 1: Core Logic — Done
- [x] `model_catalog.py` — Faster R-CNN, RetinaNet, Mask R-CNN entries
- [x] `dataset_utils.py` — COCO registration + validation
- [x] `config_builder.py` — Builds CfgNode from user params
- [x] `trainer.py` — Custom trainer with metric-emitting hook
- [x] `inferencer.py` — Wraps DefaultPredictor

### Phase 2: Workers — Done
- [x] `train_worker.py` — multiprocessing with Queue/Event IPC
- [x] `inference_worker.py` — QThread with signals

### Phase 3: GUI Shell — Done
- [x] `main_window.py` — QMainWindow + QTabWidget
- [x] `log_viewer.py` — Auto-scrolling log display
- [x] `loss_plot.py` — pyqtgraph real-time chart
- [x] `image_viewer.py` — Scrollable image with overlay support

### Phase 4: Train Tab — Done
- [x] `train_tab.py` — Full training UI with dataset/model/hyperparams/controls
- [x] `dialogs.py` — Save/load model dialogs

### Phase 5: Inference Tab — Done
- [x] `inference_tab.py` — Model loading, image/folder inference, result navigation

### Phase 6: Polish — Pending
- [ ] End-to-end testing with real dataset
- [ ] Input validation edge cases
- [ ] Error handling for training crashes
- [ ] UI styling pass

## Supported Models

| Model | Task |
|-------|------|
| Faster R-CNN (R50-FPN) | Object Detection |
| Faster R-CNN (R101-FPN) | Object Detection |
| RetinaNet (R50-FPN) | Object Detection |
| RetinaNet (R101-FPN) | Object Detection |
| Mask R-CNN (R50-FPN) | Instance Segmentation |
| Mask R-CNN (R101-FPN) | Instance Segmentation |

## Known Gotchas
1. **Detectron2 on Windows** — No official wheels; must build from source with MSVC + matching CUDA
2. **pycocotools on Windows** — Use `pycocotools-windows` or conda
3. **Multiprocessing spawn** — Windows uses spawn context; everything passed to child must be picklable
4. **Graceful stop** — Custom hook checks Event and raises SystemExit since Detectron2 has no built-in cancel
5. **CUDA context** — GUI process must not initialize CUDA during training; only the child process touches GPU

# Plan: Model Export (ONNX → TensorRT) + Deployment Inference Tab

> **Status:** Phases 1 (ONNX export), 2 (Deploy tab) and 3 (TensorRT export +
> runtime) implemented on `feature/onnx-export` (PR #3). The TensorRT paths are
> gated behind a local `tensorrt`/`pycuda` install and are experimental /
> unverified without GPU hardware.

## Context

AlchemyDetect trains/runs Detectron2 models through a PyQt6 GUI. Detectron2's
`DefaultPredictor` is convenient but slow for deployment. The user wants to
**export trained models to ONNX (and later TensorRT)** for faster inference, and
wants a **separate tab to run inference on those exported models** (kept distinct
from the existing Detectron2 Inference tab).

Confirmed scope decisions:
- **Phasing:** ONNX first; TensorRT as a follow-up (TensorRT needs an NVIDIA GPU +
  manual CUDA/cuDNN/TensorRT install, not pip-installable, no CI coverage).
- **Models:** Detection (Faster R-CNN, RetinaNet) supported first; Mask R-CNN
  (instance segmentation) exposed but marked **experimental** (TracingAdapter mask
  export is fragile).
- **Deployment:** Produce export files **and** add a separate tab that runs
  inference on ONNX/TensorRT models.

Outcome: a trained model directory (`model_final.pth` + `config.yaml` +
`class_names.json`) can be exported to `model.onnx` (+ metadata), and a new
"Deploy" tab can load and run that `.onnx` (later `.engine`) on images/folders.

## Existing pieces to reuse

- `core/inferencer.py::load_predictor(config_yaml_path, weights_path, threshold, device=None)`
  → `(DefaultPredictor, CfgNode)`; gives `predictor.model` (eval nn.Module) and the
  cfg (pixel mean/std, input format) needed for both export and runtime preprocessing.
  (A `device` override was added in Phase 1 so export can target CPU/GPU by local availability.)
- `core/inferencer.py::visualize_predictions(image_bgr, instances, metadata)` —
  reuse for drawing in the Deploy tab by constructing a Detectron2 `Instances` from
  raw ONNX/TRT outputs (detectron2 is already a hard dependency of the app).
- `workers/train_worker.py::TrainProcess` — spawn-process pattern, `mp.get_context("spawn")`,
  Queue dict-message protocol (`{"type": "log"|"status"|"device"|...}`), `daemon=False`,
  heavy imports **inside the child only**, device chosen in child. Mirror for export.
- `gui/train_tab.py` — tab layout (QGroupBox rows + button row + LogViewer), 150ms
  `QTimer` poll loop (`_poll_training`/`_handle_messages`), terminal-status handling.
- `gui/inference_tab.py` — load-model + threshold + run image/folder + results table +
  `ImageViewer` + prev/next navigation. Mirror for the Deploy tab.
- `gui/dialogs.py::load_model_dialog`, `browse_directory`, sidecar-copy pattern in
  `save_model_dialog`.
- `gui/log_viewer.py::LogViewer`, `gui/image_viewer.py::ImageViewer`.

---

## Phase 1 — ONNX export  ✅ implemented (PR #3)

### New: `alchemydetect/core/exporter.py`
Heavy imports (`torch`, `detectron2.export`, `onnx`) lazy, inside functions only.

- `is_onnx_available()` / `is_onnxruntime_available()` / `is_tensorrt_available()` —
  `importlib.util.find_spec` checks, never import, return bool.
- `resolve_model_dir(weights_path)` — validate `model_final.pth`, locate sibling
  `config.yaml` / `class_names.json`. Raise clear errors.
- `read_class_names(model_dir)` / `detect_task_from_config(config_path)` (cheap PyYAML read,
  no torch) for the GUI heads-up.
- `copy_sidecar_files(model_dir, output_dir)` — `shutil.copy2` config.yaml + class_names.json.
- `build_export_metadata(...)` → dict — pure, no torch. Records: format, opset, input_size [H,W],
  fp16, dynamic_axes, task, class names, **input/output tensor names + semantic roles**,
  pixel_mean/std + input_format (BGR/RGB), alchemydetect version, timestamp.
  This is what makes the Deploy tab self-describing.
- `export_onnx(config_path, weights_path, output_path, opset, input_size, fp16, dynamic_axes, log_fn)`:
  1. Decide device in-child (`cuda` if available else `cpu`); reuse `load_predictor(..., device=...)`;
     take `predictor.model` (eval), `cfg`.
  2. Build sample input `{"image": CHW float tensor}` on model device.
  3. Wrap `detectron2.export.TracingAdapter` (GeneralizedRCNN → `do_postprocess=False`, others direct).
  4. Run one eager pass → infer output roles by shape/dtype heuristic → set `output_names`.
  5. `torch.onnx.export(adapter, adapter.flattened_inputs, output_path, opset_version=opset,
     input_names=["image"], output_names=..., dynamic_axes=...)`.
  6. Optional fp16 via `onnxconverter_common.float16` (gated on `onnx` import).
- `validate_onnx(onnx_path, input_size, log_fn)` — optional; lazy `onnxruntime`; if absent,
  log and skip; else run one random input and log output names/shapes.
- `run_onnx_export(resolved, output_dir, options, log_fn)` — orchestrator: export → copy
  sidecars → write `export_metadata.json` → optional validate. Returns artifact paths.

### New: `alchemydetect/workers/export_worker.py` (mirrors `train_worker.py`)
- `_export_process_entry(resolved, output_dir, fmt, options, message_queue, stop_event)` —
  child entry, heavy imports inside, device decided + reported here. try/except → terminal
  `{"type":"status","status":...}` (`completed`/`stopped`/`error` with traceback as `log`).
- `class ExportProcess` — same public API as `TrainProcess`: `start(resolved, output_dir, fmt, options)`,
  `request_stop()`, `is_alive()`, `poll_metrics()`, `drain_remaining(timeout)`, `cleanup()`.
- **Message protocol**: `log`, `device`, `{"type":"artifact","path":...}` per file, `status` terminal.

### Output files (user-chosen export dir)
`model.onnx`, copied `config.yaml`, copied `class_names.json`, `export_metadata.json`.

### New: `alchemydetect/gui/export_tab.py` (mirrors `train_tab.py`)
- Model group: "Load Model…" (reuse `load_model_dialog`); info label shows class count + task;
  **flags Mask R-CNN as experimental**.
- Format group: `QComboBox ["ONNX"]` (TensorRT added in Phase 3).
- ONNX options: opset `QSpinBox` (default 17, 11–19); input H/W `QSpinBox`es (default 800×800,
  step 32); `QCheckBox` dynamic axes; `QCheckBox` fp16; `QCheckBox` "Validate (onnxruntime)"
  (disabled+tooltip if `is_onnxruntime_available()` False).
- Output dir row; button row: Export / Cancel(`request_stop`) / device label.
- Busy `QProgressBar` + `LogViewer`; 150ms `QTimer` poll (`_poll_export`, `_handle_messages`,
  terminal handling). On `completed`, QMessageBox lists written artifacts.

### Wire-in: `gui/main_window.py`
`ExportTab` added as a third tab after Inference.

---

## Phase 2 — Deployment Inference tab (run ONNX models)  ✅ implemented (PR #3)

### New: `alchemydetect/core/runtime_inferencer.py`
- `class OnnxRuntimeInferencer` — `__init__(onnx_path, metadata)` loads an
  `onnxruntime.InferenceSession` (CUDA provider if available else CPU). Methods:
  - `preprocess(image_bgr)` — resize to metadata input_size, convert BGR/RGB per
    `input_format`, apply pixel_mean/std, → CHW float tensor (reimplements Detectron2's
    `DefaultPredictor` preprocessing using values stored in `export_metadata.json`).
  - `infer(image_bgr, threshold)` — run session, read **named outputs** (boxes/classes/scores
    [/masks]) per metadata roles, scale boxes back to original size, threshold, → Detectron2
    `Instances` object.
- Returns `Instances` so the existing `visualize_predictions` + results-table code is reused unchanged.

### New: `alchemydetect/workers/deploy_inference_worker.py` (mirror `inference_worker.py`)
- `DeployInferenceWorker(QThread)` with the same signals (`result_ready(str,object,object)`,
  `progress(int,int)`, `error(str)`, `finished_all()`), `.stop()` flag, heavy imports in `run()`.
  Uses `OnnxRuntimeInferencer` (later branches to TensorRT). Reuse
  `InferenceWorker.collect_image_paths` for folders.

### New: `alchemydetect/gui/deploy_tab.py` (mirror `inference_tab.py`)
- Load Model dialog accepts `*.onnx` (Phase 3: `*.engine`); auto-load sibling
  `export_metadata.json` + `class_names.json`. Threshold spin, Run Image/Folder/Stop,
  progress bar, `ImageViewer`, detections `QTableWidget`, prev/next nav — reuse the
  inference_tab structure and `_show_result`/`_update_nav` logic.
- Wire into `main_window.py` as a fourth tab ("Deploy").

---

## Phase 3 — TensorRT (export + runtime), gated/optional  ✅ implemented (PR #3)

- `exporter.export_tensorrt(onnx_path, engine_path, fp16, workspace_gb, log_fn)` — ensure ONNX
  exists (build via `export_onnx` if user picks TensorRT directly); `import tensorrt as trt`,
  build `Builder`/EXPLICIT_BATCH network/`OnnxParser`, set workspace + FP16 flag (if
  `platform_has_fast_fp16`), `build_serialized_network`, write `.engine`. Log parser errors clearly.
- `export_worker._export_process_entry` branches on `fmt == "tensorrt"`.
- `export_tab`: TensorRT appears in format combo only if `is_tensorrt_available()` else disabled
  w/ tooltip "Install NVIDIA TensorRT"; options: fp16, workspace GB. Labeled "advanced, GPU required".
- `runtime_inferencer.TensorRTInferencer` + `deploy_inference_worker` branch: load engine, run via
  TRT execution context (+ CUDA buffers), same `Instances` output → reuses Deploy tab unchanged.
- Never hard-import `tensorrt` at module/GUI load.

---

## Packaging
- `pyproject.toml` `[project.optional-dependencies]`: ✅ added
  `export = ["onnx>=1.15", "onnxruntime>=1.17", "onnxconverter-common>=1.14"]`. TensorRT kept out
  (documented as user-installed). Optional future `export-gpu = ["onnxruntime-gpu"]` (note mutual
  exclusivity with onnxruntime).
- `requirements.txt`: ✅ commented note — `pip install alchemydetect[export]` for ONNX; TensorRT
  installed manually matching CUDA/cuDNN.

## Testing (CI is CPU-only, detectron2 from source; keep torch/d2 out of importable units)
`tests/test_exporter.py`, `tests/test_export_worker.py` (✅ Phase 1),
`tests/test_runtime_inferencer.py` (Phase 2), plus Qt offscreen smoke tests:
- `build_export_metadata` key/passthrough; `copy_sidecar_files`; `resolve_model_dir` (auto-detect +
  missing-pth error); `read_class_names`; `detect_task_from_config` (PyYAML); `is_*_available()`
  return bools & never raise. ✅
- `ExportProcess` lifecycle without start (mirror `tests/test_train_worker.py`). ✅
- `OnnxRuntimeInferencer.preprocess` shape/normalization on a synthetic image + fake metadata (no
  real session); postprocess raw-arrays→Instances mapping with a stub. (Phase 2)
- Widget smoke tests: `ExportTab` (✅) / `DeployTab` instantiate; format combo populated;
  onnxruntime checkbox disabled when unavailable.
- Manual/not-CI: real ONNX export per model family, onnxruntime numerical validation, all TensorRT
  paths. Respect ruff (line-length 120, E/F/W/I).

## Docs
- README.md — ✅ Export feature + usage + `pip install alchemydetect[export]`; Deploy added in Phase 2.
- CHANGELOG.md — ✅ `### Added` under `[Unreleased]`.
- INSTALL.md — ✅ "Optional: Model Export" (ONNX extra; manual TensorRT/CUDA notes, Windows caveat).
- plan.md — ✅ Phase 7 section + export gotchas.

## Risks
- Mask R-CNN ONNX export fragile → detection-first, segmentation experimental, surface tracing
  tracebacks via the `error` status.
- ONNX output ordering/schema → pin `output_names`, persist semantic roles in metadata; Deploy tab
  reads named outputs.
- opset/torch/detectron2 coupling → default opset 17, record versions, document tested matrix.
- Keep all torch/CUDA in the child process (repo already had a GUI-CUDA bug — see config_builder.py).
- TensorRT availability/install pain → fully gated; `.engine` files are not portable across
  GPU/TRT versions (document, store TRT version in metadata).
- Cancel can't interrupt the single opaque `torch.onnx.export` call mid-trace → rely on process
  isolation; document that Cancel takes effect at stage boundaries.
- onnxruntime vs onnxruntime-gpu conflict → document mutual exclusivity.

## Verification (end-to-end, manual)
1. `pip install -e ".[dev,export]"` (with torch + detectron2 installed).
2. Train a tiny Faster R-CNN (existing Train tab) → save model dir.
3. Export tab → load that dir → ONNX, opset 17, 800×800, validate-with-onnxruntime → confirm
   `model.onnx` + sidecars + `export_metadata.json` written and validation logs output shapes.
4. Deploy tab → load `model.onnx` → run on a folder → boxes/scores/classes match the Detectron2
   Inference tab within tolerance. (Phase 2)
5. (GPU machine, Phase 3) Export → TensorRT `.engine`; Deploy tab runs it; compare speed/results.
6. `python -m pytest tests/ -q` and `ruff check` / `ruff format --check` green.

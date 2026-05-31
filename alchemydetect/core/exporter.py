"""Export trained Detectron2 models to deployment formats (ONNX, later TensorRT).

The heavy work (torch / detectron2 / onnx imports) is done lazily inside the
functions that need it, so the pure helpers — availability checks, path
resolution, sidecar copying and metadata building — stay importable (and
testable) on machines without torch or detectron2 installed.
"""

import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

# Bake a low score threshold into the exported model so the deployment runtime
# can apply the user's threshold freely; a high baked threshold would otherwise
# be a floor the runtime could never drop below.
EXPORT_SCORE_THRESH = 0.05

WEIGHTS_FILENAME = "model_final.pth"
CONFIG_FILENAME = "config.yaml"
CLASS_NAMES_FILENAME = "class_names.json"
METADATA_FILENAME = "export_metadata.json"
ONNX_FILENAME = "model.onnx"
ENGINE_FILENAME = "model.engine"


# --------------------------------------------------------------------------- #
# Availability checks (no imports of the heavy/optional packages themselves)
# --------------------------------------------------------------------------- #
def is_onnx_available():
    """Return True if the `onnx` package is importable."""
    return importlib.util.find_spec("onnx") is not None


def is_onnxruntime_available():
    """Return True if the `onnxruntime` package is importable."""
    return importlib.util.find_spec("onnxruntime") is not None


def is_tensorrt_available():
    """Return True if the `tensorrt` package is importable."""
    return importlib.util.find_spec("tensorrt") is not None


# --------------------------------------------------------------------------- #
# Path resolution and sidecar handling (pure filesystem, no torch)
# --------------------------------------------------------------------------- #
def resolve_model_dir(weights_path):
    """Resolve a model directory from a chosen weights file.

    Args:
        weights_path: Path to a ``.pth`` weights file (typically model_final.pth).

    Returns:
        Dict with keys ``model_dir``, ``weights_path``, ``config_path`` and
        ``class_names_path`` (the last is None if no class_names.json sits next
        to the weights).

    Raises:
        FileNotFoundError: If the weights file or its sibling config.yaml is missing.
    """
    weights = Path(weights_path)
    if not weights.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights}")

    model_dir = weights.parent
    config_path = model_dir / CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No {CONFIG_FILENAME} found next to the weights in {model_dir}. "
            "Export needs the Detectron2 config that was saved during training."
        )

    class_names_path = model_dir / CLASS_NAMES_FILENAME
    return {
        "model_dir": str(model_dir),
        "weights_path": str(weights),
        "config_path": str(config_path),
        "class_names_path": str(class_names_path) if class_names_path.is_file() else None,
    }


def read_class_names(model_dir):
    """Read class_names.json from a model directory, or [] if absent/invalid."""
    path = Path(model_dir) / CLASS_NAMES_FILENAME
    if not path.is_file():
        return []
    try:
        with open(path, "r") as f:
            names = json.load(f)
        return names if isinstance(names, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def detect_task_from_config(config_path):
    """Cheaply detect the task from a saved config.yaml without importing torch.

    Returns "instance_segmentation", "detection", or "unknown" (if PyYAML is
    unavailable or the file can't be parsed). The authoritative task is recorded
    in export metadata at export time; this is only a GUI heads-up.
    """
    try:
        import yaml
    except ImportError:
        return "unknown"
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return "instance_segmentation" if cfg.get("MODEL", {}).get("MASK_ON") else "detection"
    except (OSError, yaml.YAMLError, AttributeError):
        return "unknown"


def copy_sidecar_files(model_dir, output_dir):
    """Copy config.yaml and class_names.json from model_dir into output_dir.

    Returns:
        List of destination paths that were actually copied.
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for filename in (CONFIG_FILENAME, CLASS_NAMES_FILENAME):
        src = model_dir / filename
        if src.is_file():
            dest = output_dir / filename
            shutil.copy2(src, dest)
            copied.append(str(dest))
    return copied


def _alchemydetect_version():
    """Return the installed alchemydetect version, or 'unknown'."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("alchemydetect")
        except PackageNotFoundError:
            return "unknown"
    except ImportError:
        return "unknown"


def build_export_metadata(
    *,
    model_format,
    opset,
    input_size,
    fp16,
    dynamic_axes,
    task,
    class_names,
    output_names,
    output_roles,
    preprocessing,
    timestamp=None,
):
    """Build a serializable metadata dict describing an exported model.

    This is pure (no torch / detectron2) so it can be unit tested. It records
    everything the deployment runtime needs to reproduce preprocessing and to
    map the model's output tensors back to boxes / scores / classes / masks.

    Args:
        model_format: "onnx" or "tensorrt".
        opset: ONNX opset version used.
        input_size: (height, width) the model was traced with.
        fp16: Whether fp16 conversion was applied.
        dynamic_axes: Whether dynamic height/width/detection axes were enabled.
        task: "detection" or "instance_segmentation".
        class_names: List of class name strings (ordered by contiguous class id).
        output_names: Ordered ONNX output tensor names.
        output_roles: Semantic role per output ("boxes"/"scores"/"classes"/"masks"/"unknown").
        preprocessing: Dict with pixel_mean, pixel_std, input_format, min_size, max_size.
        timestamp: Optional ISO timestamp; defaults to now (UTC).
    """
    height, width = input_size
    return {
        "format": model_format,
        "opset": opset,
        "input_size": [int(height), int(width)],
        "fp16": bool(fp16),
        "dynamic_axes": bool(dynamic_axes),
        "task": task,
        "class_names": list(class_names),
        "input_name": "image",
        "output_names": list(output_names),
        "output_roles": list(output_roles),
        "preprocessing": preprocessing,
        "alchemydetect_version": _alchemydetect_version(),
        "created": timestamp or datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
# Heavy export path (torch / detectron2 / onnx imported lazily inside)
# --------------------------------------------------------------------------- #
def _infer_output_roles(outputs):
    """Heuristically map traced output tensors to semantic roles by shape/dtype.

    Detectron2 flattens an Instances object into a tuple of tensors whose order
    depends on field insertion order, so we identify each tensor by its shape:
    boxes (N, 4), classes (N,) integer, scores (N,) float, masks (N, ..., H, W).

    Returns:
        (names, roles) parallel lists.
    """
    import torch

    names, roles = [], []
    for i, t in enumerate(outputs):
        if not hasattr(t, "dim"):
            names.append(f"output_{i}")
            roles.append("unknown")
            continue
        if t.dim() == 2 and t.shape[-1] == 4:
            names.append("pred_boxes")
            roles.append("boxes")
        elif t.dim() == 1 and not torch.is_floating_point(t):
            names.append("pred_classes")
            roles.append("classes")
        elif t.dim() == 1:
            names.append("scores")
            roles.append("scores")
        elif t.dim() >= 3:
            names.append("pred_masks")
            roles.append("masks")
        else:
            names.append(f"output_{i}")
            roles.append("unknown")
    return names, roles


def _build_preprocessing(cfg):
    """Extract the preprocessing parameters the runtime needs from a cfg."""
    return {
        "pixel_mean": [float(x) for x in cfg.MODEL.PIXEL_MEAN],
        "pixel_std": [float(x) for x in cfg.MODEL.PIXEL_STD],
        "input_format": cfg.INPUT.FORMAT,
        "min_size": int(cfg.INPUT.MIN_SIZE_TEST),
        "max_size": int(cfg.INPUT.MAX_SIZE_TEST),
    }


def export_onnx(config_path, weights_path, output_path, opset, input_size, fp16, dynamic_axes, log_fn):
    """Trace a Detectron2 model and export it to ONNX.

    Mirrors detectron2's official tools/deploy/export_model.py tracing route:
    GeneralizedRCNN models are wrapped so inference runs without postprocessing,
    other meta-architectures (e.g. RetinaNet) are traced directly.

    Args:
        config_path: Saved config.yaml path.
        weights_path: .pth weights path.
        output_path: Where to write model.onnx.
        opset: ONNX opset version.
        input_size: (height, width) to trace with.
        fp16: If True, convert the ONNX graph to fp16 (requires onnx package).
        dynamic_axes: If True, mark height/width and detection count as dynamic.
        log_fn: Callable(str) for progress messages.

    Returns:
        Dict with output_path, output_names, output_roles, task, preprocessing.

    Raises:
        RuntimeError: If the `onnx` package is not installed (torch.onnx.export
            requires it to serialize the graph).
    """
    if not is_onnx_available():
        raise RuntimeError(
            "ONNX export requires the 'onnx' package, which is not installed.\n"
            "Install the export extras with:  pip install alchemydetect[export]"
        )

    import torch
    from detectron2.export import TracingAdapter
    from detectron2.modeling import GeneralizedRCNN

    from alchemydetect.core.inferencer import load_predictor

    # Decide the device here, in the child process, never in the GUI process.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_fn(f"Loading model and config (device: {device})...")
    predictor, cfg = load_predictor(config_path, weights_path, threshold=EXPORT_SCORE_THRESH, device=device)
    model = predictor.model
    model.eval()

    task = "instance_segmentation" if cfg.MODEL.MASK_ON else "detection"
    preprocessing = _build_preprocessing(cfg)

    height, width = input_size
    # A representative raw image in the model's expected (C, H, W) 0-255 range;
    # the model applies normalization/padding internally during tracing.
    sample = torch.rand(3, height, width, device=device) * 255.0
    sample_inputs = [{"image": sample}]

    if isinstance(model, GeneralizedRCNN):
        log_fn("Wrapping GeneralizedRCNN for tracing (do_postprocess=False)...")

        def inference(model, inputs):
            instances = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": instances}]

        adapter = TracingAdapter(model, sample_inputs, inference)
    else:
        log_fn(f"Tracing meta-arch '{cfg.MODEL.META_ARCHITECTURE}' directly...")
        adapter = TracingAdapter(model, sample_inputs)

    adapter.eval()

    log_fn("Running a trace pass to determine output layout...")
    with torch.no_grad():
        eager_outputs = adapter(*adapter.flattened_inputs)
    if not isinstance(eager_outputs, (tuple, list)):
        eager_outputs = (eager_outputs,)
    output_names, output_roles = _infer_output_roles(eager_outputs)
    log_fn(f"Outputs: {list(zip(output_names, output_roles))}")

    axes = None
    if dynamic_axes:
        axes = {"image": {1: "height", 2: "width"}}
        for name in output_names:
            axes[name] = {0: "num_detections"}

    log_fn(f"Exporting ONNX (opset {opset})...")
    with torch.no_grad():
        torch.onnx.export(
            adapter,
            adapter.flattened_inputs,
            output_path,
            opset_version=opset,
            input_names=["image"],
            output_names=output_names,
            dynamic_axes=axes,
        )
    log_fn(f"Wrote {output_path}")

    if fp16:
        _convert_fp16(output_path, log_fn)

    return {
        "output_path": output_path,
        "output_names": output_names,
        "output_roles": output_roles,
        "task": task,
        "preprocessing": preprocessing,
    }


def _convert_fp16(onnx_path, log_fn):
    """Convert an ONNX model to fp16 in place, if the tooling is available."""
    if not is_onnx_available():
        log_fn("onnx package not installed; skipping fp16 conversion.")
        return
    try:
        import onnx
        from onnxconverter_common import float16

        log_fn("Converting ONNX graph to fp16...")
        model = onnx.load(onnx_path)
        model = float16.convert_float_to_fp16(model)
        onnx.save(model, onnx_path)
        log_fn("fp16 conversion complete.")
    except ImportError:
        log_fn("onnxconverter-common not installed; skipping fp16 conversion.")


def validate_onnx(onnx_path, input_size, log_fn):
    """Optionally run one random input through onnxruntime to sanity-check the graph.

    Returns:
        Dict of {output_name: shape} on success, or None if onnxruntime is absent.
    """
    if not is_onnxruntime_available():
        log_fn("onnxruntime not installed; skipping validation.")
        return None

    import numpy as np
    import onnxruntime as ort

    log_fn("Validating with onnxruntime...")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    height, width = input_size
    dummy = (np.random.rand(3, height, width) * 255.0).astype(np.float32)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: dummy})

    shapes = {}
    for meta, arr in zip(session.get_outputs(), outputs):
        shapes[meta.name] = list(arr.shape)
        log_fn(f"  {meta.name}: {list(arr.shape)}")
    return shapes


def run_onnx_export(resolved, output_dir, options, log_fn):
    """Orchestrate a full ONNX export: trace, write metadata, copy sidecars, validate.

    Args:
        resolved: Dict from resolve_model_dir (model_dir, weights_path, config_path, ...).
        output_dir: Destination directory for export artifacts.
        options: Dict with opset, input_size (h, w), fp16, dynamic_axes, validate.
        log_fn: Callable(str) for progress messages.

    Returns:
        List of artifact file paths written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = str(output_dir / ONNX_FILENAME)
    result = export_onnx(
        config_path=resolved["config_path"],
        weights_path=resolved["weights_path"],
        output_path=onnx_path,
        opset=options["opset"],
        input_size=options["input_size"],
        fp16=options["fp16"],
        dynamic_axes=options["dynamic_axes"],
        log_fn=log_fn,
    )

    artifacts = [onnx_path]
    artifacts.extend(copy_sidecar_files(resolved["model_dir"], output_dir))

    metadata = build_export_metadata(
        model_format="onnx",
        opset=options["opset"],
        input_size=options["input_size"],
        fp16=options["fp16"],
        dynamic_axes=options["dynamic_axes"],
        task=result["task"],
        class_names=read_class_names(resolved["model_dir"]),
        output_names=result["output_names"],
        output_roles=result["output_roles"],
        preprocessing=result["preprocessing"],
    )
    metadata_path = str(output_dir / METADATA_FILENAME)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log_fn(f"Wrote {metadata_path}")
    artifacts.append(metadata_path)

    if options.get("validate"):
        validate_onnx(onnx_path, options["input_size"], log_fn)

    return artifacts


# --------------------------------------------------------------------------- #
# TensorRT export (ONNX -> engine), gated behind a local TensorRT install
# --------------------------------------------------------------------------- #
def export_tensorrt(onnx_path, engine_path, fp16, workspace_gb, input_size, log_fn):
    """Build a serialized TensorRT engine from an ONNX model.

    Args:
        onnx_path: Path to an existing model.onnx.
        engine_path: Where to write the serialized engine.
        fp16: Build with FP16 if the platform supports it.
        workspace_gb: Builder workspace memory pool limit, in GB.
        input_size: (height, width) used as the optimization profile's opt shape
            when the ONNX input has dynamic spatial dimensions.
        log_fn: Callable(str) for progress messages.

    Raises:
        RuntimeError: If TensorRT is unavailable, ONNX parsing fails, or the
            engine build returns nothing.
    """
    if not is_tensorrt_available():
        raise RuntimeError(
            "TensorRT export requires the 'tensorrt' package, which is not installed.\n"
            "TensorRT is not pip-installable from this project — install it manually to "
            "match your CUDA/cuDNN versions (see INSTALL.md)."
        )

    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    log_fn("Parsing ONNX into a TensorRT network...")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"TensorRT failed to parse ONNX: {errors}")

    config = builder.create_builder_config()
    workspace_bytes = int(workspace_gb * (1 << 30))
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    except AttributeError:  # older TensorRT
        config.max_workspace_size = workspace_bytes

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            log_fn("FP16 enabled.")
        else:
            log_fn("Platform has no fast FP16 support; building in FP32.")

    # Dynamic input dims need an optimization profile (min/opt/max shapes).
    inp = network.get_input(0)
    dims = list(inp.shape)
    if any(d == -1 for d in dims):
        channels = dims[0] if dims[0] != -1 else 3
        height, width = input_size
        profile = builder.create_optimization_profile()
        profile.set_shape(
            inp.name,
            (channels, 256, 256),
            (channels, height, width),
            (channels, max(height, 1344), max(width, 1344)),
        )
        config.add_optimization_profile(profile)
        log_fn(f"Added optimization profile for dynamic input '{inp.name}'.")

    log_fn("Building TensorRT engine (this can take several minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed (build_serialized_network returned None).")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    log_fn(f"Wrote {engine_path}")
    return engine_path


def run_tensorrt_export(resolved, output_dir, options, log_fn):
    """Orchestrate a TensorRT export: build the ONNX first, then the engine.

    The intermediate model.onnx and the shared export_metadata.json are kept so
    the Deploy tab can run either artifact.

    Returns:
        List of artifact file paths written (onnx, sidecars, metadata, engine).
    """
    if not is_tensorrt_available():
        raise RuntimeError(
            "TensorRT export requires the 'tensorrt' package, which is not installed.\n"
            "Install it manually to match your CUDA/cuDNN versions (see INSTALL.md)."
        )

    output_dir = Path(output_dir)
    log_fn("Step 1/2 — exporting ONNX...")
    artifacts = run_onnx_export(resolved, output_dir, options, log_fn)

    log_fn("Step 2/2 — building TensorRT engine...")
    engine_path = export_tensorrt(
        onnx_path=str(output_dir / ONNX_FILENAME),
        engine_path=str(output_dir / ENGINE_FILENAME),
        fp16=options["fp16"],
        workspace_gb=options.get("workspace_gb", 4.0),
        input_size=options["input_size"],
        log_fn=log_fn,
    )
    artifacts.append(engine_path)
    return artifacts

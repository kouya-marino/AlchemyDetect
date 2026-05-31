# TensorRT in AlchemyDetect — How it works + install guide

This document explains how the TensorRT path works and gives exact install steps
for getting a working TensorRT stack so you can **export** `.engine` files (Export
tab) and **run** them (Deploy tab).

> **TensorRT runs only on NVIDIA GPUs.** There is no CPU, AMD, Intel-GPU, or Apple
> Silicon build. macOS is not supported. If you don't have an NVIDIA GPU, use the
> ONNX (onnxruntime) path instead.

---

## 1. How the TensorRT model works

TensorRT is NVIDIA's inference-only optimizer + runtime. You don't train with it —
you take an already-trained model and TensorRT compiles it into a hardware-specific
**engine** that runs the forward pass much faster on that exact GPU.

In AlchemyDetect the pipeline is:

```
model_final.pth  ->  model.onnx  ->  model.engine  ->  GPU inference
   (PyTorch)        (TracingAdapter)  (TensorRT build)   (TensorRT runtime)
```

There are two distinct phases: **build** (slow, one-time) and **runtime** (fast,
per image).

### Phase 1 — Build the engine (`core/exporter.py::export_tensorrt`)

Triggered from the Export tab when you choose the "TensorRT" format. It first
produces `model.onnx`, then:

1. **Parse** — `trt.OnnxParser` reads the ONNX graph into a TensorRT network.
2. **Optimize** — `builder.build_serialized_network(...)` applies:
   - **Layer & tensor fusion** — merges e.g. conv+bias+ReLU into one kernel and
     removes intermediate GPU-memory round-trips (usually the biggest win).
   - **Precision reduction** — with the FP16 flag, math runs in half precision:
     ~2× throughput, half the memory, negligible accuracy loss for detection.
     (INT8 is faster still but needs calibration data — not done here.)
   - **Kernel auto-tuning** — benchmarks several CUDA kernels *on your actual GPU*
     and keeps the fastest. This is why the build takes minutes and why the result
     is tied to your specific card.
   - **Memory planning** — reuses a single workspace pool (the "workspace GB"
     option) and plans tensor lifetimes to minimize allocations.
3. **Dynamic shapes** — because the ONNX is exported with dynamic height/width,
   TensorRT needs an **optimization profile** (min/opt/max input sizes). The
   exporter adds one (opt = your chosen input size).
4. **Serialize** — the tuned plan is written to `model.engine` (a binary blob).

> The `.engine` is **not portable**: it is specialized to your GPU model +
> TensorRT version + CUDA version. Moving it to a different GPU/TensorRT will fail
> to deserialize — rebuild it per machine.

### Phase 2 — Runtime inference (`core/runtime_inferencer.py::TensorRTInferencer`)

Triggered from the Deploy tab when you load a `.engine`:

1. **Deserialize** — `runtime.deserialize_cuda_engine(bytes)`; `create_execution_context()`.
2. **Preprocess** (shared with the ONNX path) — ResizeShortestEdge to the model's
   input format, CHW float32; the model normalizes internally.
3. **Bind buffers** — allocate GPU memory with `pycuda`, copy input host→device,
   register each tensor address; `set_input_shape` for the dynamic input.
4. **Execute** — `execute_async_v3(stream)` runs the whole optimized graph in one
   launch; `stream.synchronize()` waits.
5. **Postprocess** (shared) — copy outputs device→host, map them to
   boxes/scores/classes by the roles recorded in `export_metadata.json`, scale
   boxes back to the original image, and build a Detectron2 `Instances` (so the
   existing visualizer and detections table are reused).

### Why it's fast

- **GPU, not CPU** — the first and biggest factor.
- **Fused kernels + FP16 + GPU-specific tuning** — beyond generic PyTorch or
  onnxruntime-GPU; this is what genuinely beats a PyTorch GPU `.pth` (often 2–5×).
- **No Python per-op overhead** — one compiled artifact, not Python-dispatched ops.

---

## 2. Prerequisites (read first)

TensorRT, CUDA, cuDNN, and your framework (PyTorch / onnxruntime-gpu) must all
agree on the **same CUDA major version**. Mismatches are the #1 source of "DLL
missing" / "cannot load" errors.

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA, with a recent driver (`nvidia-smi` works) |
| CUDA Toolkit | e.g. 12.x (recommended) or 11.8 |
| cuDNN | matching the CUDA major (cuDNN 9.x for CUDA 12, cuDNN 8.x for CUDA 11) |
| TensorRT | a version that supports your CUDA (TensorRT 10.x → CUDA 12.x/11.8) |
| pycuda | required only to **run** engines in the Deploy tab |

Always confirm exact combinations against the official support matrix:
https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html

> **AlchemyDetect note:** if your PyTorch is the CUDA 11.8 build, prefer a CUDA-11
> TensorRT/onnxruntime stack so everything matches; if you move to CUDA 12,
> reinstall PyTorch as `cu12x` too.

---

## 3. Install — easiest method (pip wheels): Linux & Windows

For TensorRT 8.6+ / 10.x, NVIDIA publishes pip wheels that bundle the needed CUDA
libraries. This is the simplest cross-platform route and is usually enough for the
AlchemyDetect Export/Deploy paths.

```bash
# Activate the same virtual environment AlchemyDetect uses, then:
python -m pip install --upgrade pip

# CUDA 12.x systems:
python -m pip install tensorrt              # pulls tensorrt-cu12 wheels

# CUDA 11.x systems (pin the cu11 variant):
python -m pip install tensorrt-cu11

# If pip can't find it, add NVIDIA's index:
python -m pip install tensorrt --extra-index-url https://pypi.nvidia.com
```

Then install pycuda (needed to **run** engines):

```bash
python -m pip install pycuda
```

Verify (see §6). If `import tensorrt` works and reports a version, you're done —
skip the manual methods below.

> pycuda builds a small C extension. On **Windows** it needs the MSVC C++ build
> tools and the CUDA Toolkit (`nvcc`) on PATH; on **Linux** it needs `gcc` and the
> CUDA Toolkit. If pycuda is painful, `cuda-python` is an alternative buffer
> backend (the app currently targets pycuda).

---

## 4. Install — manual method (full control)

Use this when you need a specific TensorRT version, GPU-wide availability (not just
one venv), or the pip wheels don't fit your setup.

### 4a. Linux (tar)

1. Install the **CUDA Toolkit** and **cuDNN** for your CUDA version (NVIDIA repos
   or runfiles). Confirm `nvcc --version` and `nvidia-smi`.
2. Download the **TensorRT tar** matching your CUDA from
   https://developer.nvidia.com/tensorrt (NVIDIA login required).
3. Extract and expose the libraries:
   ```bash
   tar -xzvf TensorRT-10.x.x.x.Linux.x86_64-gnu.cuda-12.x.tar.gz
   export TRT_HOME=$PWD/TensorRT-10.x.x.x
   export LD_LIBRARY_PATH=$TRT_HOME/lib:$LD_LIBRARY_PATH
   ```
   (persist the exports in `~/.bashrc`).
4. Install the bundled Python wheels into your venv (pick your Python version):
   ```bash
   python -m pip install $TRT_HOME/python/tensorrt-*-cp310-*.whl
   python -m pip install pycuda
   ```

### 4b. Windows (zip)

1. Install the **CUDA Toolkit** and **cuDNN** matching your CUDA version; ensure
   their `bin` folders are on `PATH`. Confirm `nvidia-smi` and `nvcc --version`.
2. Download the **TensorRT zip** for your CUDA from
   https://developer.nvidia.com/tensorrt and extract to e.g. `C:\TensorRT-10.x`.
3. Add `C:\TensorRT-10.x\lib` to your system **PATH** (so the `.dll`s load), then
   reopen your shell.
4. Install the bundled wheel + pycuda into your venv:
   ```powershell
   python -m pip install C:\TensorRT-10.x\python\tensorrt-*-cp310-*.whl
   python -m pip install pycuda
   ```
   (pycuda on Windows needs the **MSVC C++ build tools** and CUDA `nvcc` on PATH.)

### 4c. Docker (Linux, no host install)

The fastest clean environment — NVIDIA's containers ship CUDA + cuDNN + TensorRT:

```bash
docker run --gpus all -it --rm \
  -v "$PWD":/workspace -w /workspace \
  nvcr.io/nvidia/tensorrt:24.xx-py3
# then: pip install -e ".[export]" pycuda  (PyTorch/Detectron2 as needed)
```
Requires the NVIDIA Container Toolkit on the host.

---

## 5. Unsupported platforms

- **macOS** — no TensorRT. Use the ONNX path (CPU, or onnxruntime on supported GPUs).
- **AMD / Intel GPUs** — no TensorRT. Use ONNX.
- **CPU-only** — no TensorRT. Use the ONNX path (`pip install alchemydetect[export]`).

---

## 6. Verify the install

```bash
python -c "import tensorrt as trt; print('TensorRT', trt.__version__)"
python -c "import pycuda.autoinit, pycuda.driver as cuda; print('CUDA devices:', cuda.Device.count())"
python -c "import torch; print('torch CUDA:', torch.version.cuda, torch.cuda.is_available())"
```

All three should succeed and report consistent CUDA versions. In AlchemyDetect:

- The **Export** tab will now show a **TensorRT** format option.
- The **Deploy** tab can load a `.engine`; its side panel shows
  `Runtime: TensorRT` and the per-image detection time.

---

## 7. Use it in AlchemyDetect

1. Install the ONNX extras too (TensorRT export builds the ONNX first):
   `pip install alchemydetect[export]`
2. **Export tab** → Load Model → choose **TensorRT** → set FP16 / workspace /
   input size / score threshold → Export. You'll get `model.onnx`, `model.engine`,
   `config.yaml`, `class_names.json`, and `export_metadata.json`.
3. **Deploy tab** → Load Model → select `model.engine` → Run on Image/Folder.
   Confirm the side panel reads `Runtime: TensorRT`.

---

## 8. Running the ONNX path on GPU (onnxruntime-gpu)

TensorRT gives the biggest speedup, but you can also accelerate the plain **ONNX**
Deploy path on a GPU with `onnxruntime-gpu` — no TensorRT needed. The catch: the
`onnxruntime-gpu` build must match your installed CUDA + cuDNN, or it logs
`cublasLt64_*.dll is missing` and silently falls back to CPU (the Deploy side panel
will then show `CPUExecutionProvider`, and inference is slow).

### Version matching (the crucial part)

| onnxruntime-gpu | CUDA | cuDNN |
|-----------------|------|-------|
| `1.17.x`        | 11.8 | 8.x   |
| `1.18`–`1.20+`  | 12.x | 9.x   |

Pick the build that matches the CUDA your PyTorch uses — check with:

```bash
python -c "import torch; print(torch.version.cuda)"
```

> Install only **one** of `onnxruntime` (CPU) or `onnxruntime-gpu` — having both in
> the same environment causes import conflicts.

### If your CUDA is 11.8 (e.g. PyTorch `cu118`)

```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.17.1
```

Install **CUDA 11.8** + **cuDNN 8.x** and put their `bin` directories on `PATH`.
PyTorch bundles its own CUDA libraries and does **not** expose them to onnxruntime,
so onnxruntime needs a real CUDA/cuDNN install visible on `PATH`.

### If your CUDA is 12.x

```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu        # recent builds target CUDA 12 + cuDNN 9
```

Install **CUDA 12.x** + **cuDNN 9.x** on `PATH`. (If you're on CUDA 11.8 PyTorch,
switching to CUDA 12 means also reinstalling PyTorch as a `cu12x` build.)

### Verify

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

`CUDAExecutionProvider` should be listed. Then in the **Deploy** tab the side panel
reads `Runtime: onnxruntime — CUDAExecutionProvider` instead of `CPUExecutionProvider`.

> Even on GPU, onnxruntime is roughly on par with PyTorch-GPU for detection models;
> for a genuine speedup over the `.pth`, use TensorRT (sections above).

---

## 9. Troubleshooting

- **`Could not load ... cublasLt64_12.dll` / `cudnn... is missing`** — your CUDA/
  cuDNN runtime doesn't match the TensorRT (or onnxruntime-gpu) build. Align the
  CUDA major version across CUDA Toolkit, cuDNN, TensorRT, and PyTorch.
- **`Failed to deserialize the TensorRT engine`** — the engine was built on a
  different GPU or TensorRT version. Engines are not portable; **rebuild** on the
  target machine.
- **Engine build fails / runs out of memory** — lower the workspace size, reduce
  the input size, or free GPU memory.
- **pycuda build/install fails** — install the CUDA Toolkit (`nvcc`) and a C++
  compiler (MSVC on Windows, gcc on Linux); ensure both are on PATH.
- **TensorRT 8 vs 10 API** — AlchemyDetect targets the TensorRT 10 Python API
  (`num_io_tensors`, `set_input_shape`, `execute_async_v3`). On TensorRT 8.x some
  calls differ and the runtime may need adjustment.

> **Status:** the TensorRT export and runtime in AlchemyDetect are implemented but
> not yet verified on real hardware. Expect to validate (and possibly tweak)
> against your specific CUDA/TensorRT versions.

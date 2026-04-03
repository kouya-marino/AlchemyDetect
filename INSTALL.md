# Installation Guide

## Prerequisites

- **Python 3.10 or 3.11** (other versions are not supported due to Detectron2/PyTorch compatibility)
- **NVIDIA GPU with CUDA support** (recommended; CPU-only mode is supported but slow)

### Platform-specific requirements

| Platform | Additional Requirements |
|----------|----------------------|
| **Windows** | CUDA Toolkit 11.8, MSVC Build Tools 2019+ (for compiling Detectron2) |
| **Linux** | CUDA Toolkit 11.8 (optional — prebuilt Detectron2 wheels available), `gcc/g++` |
| **macOS** | Xcode Command Line Tools (`xcode-select --install`). GPU training not supported (CPU only) |

## Step-by-Step Setup

### 1. Create a virtual environment

```bash
python -m venv venv

# Activate:
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 2. Install PyTorch

Visit https://pytorch.org/get-started/locally/ for the exact command matching your platform and CUDA version.

**Windows / Linux with CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Windows / Linux with CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**macOS (CPU only):**
```bash
pip install torch torchvision
```

**Linux / Windows CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install pycocotools

**Linux / macOS:**
```bash
pip install pycocotools
```

**Windows** — the standard package may fail to build. Try these in order:
```bash
# Option A — pip (requires MSVC)
pip install pycocotools

# Option B — conda
conda install -c conda-forge pycocotools
```

### 4. Install build tools

```bash
pip install --upgrade pip setuptools wheel ninja pyproject-toml
```

### 5. Install Detectron2

#### Linux (easiest)

Prebuilt wheels are available for Linux:
```bash
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

If no prebuilt wheel matches your PyTorch version, fall back to source:
```bash
pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git
```

#### Windows

Must build from source. Run from a **Developer Command Prompt** (or ensure `cl.exe` is in PATH):
```bash
pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git
```

**Important:** The `--no-build-isolation` flag is required so the build can access PyTorch.

If the build fails:
- Ensure CUDA Toolkit version matches your PyTorch build (e.g. CUDA 11.8 for cu118)
- Ensure `cl.exe` is in your PATH — install Visual Studio Build Tools with "Desktop development with C++"
- If you have multiple CUDA versions installed, set `CUDA_HOME`:
  ```bash
  set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
  ```

#### macOS

Build from source (CPU only — no CUDA on macOS):
```bash
CC=clang CXX=clang++ FORCE_CUDA=0 pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git
```

Requires Xcode Command Line Tools:
```bash
xcode-select --install
```

### 6. Install remaining dependencies

```bash
pip install PyQt6 pyqtgraph opencv-python Pillow numpy
```

Or install everything (except PyTorch and Detectron2) from requirements.txt:
```bash
pip install -r requirements.txt
```

### 7. Verify installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import detectron2; print('Detectron2:', detectron2.__version__)"
python -c "from PyQt6 import QtWidgets; print('PyQt6: OK')"
```

### 8. Run the application

```bash
python main.py
```

## Troubleshooting

### Detectron2 build fails on Windows
- Make sure you're using Python 3.10 or 3.11
- Install Visual Studio Build Tools with "Desktop development with C++"
- Run the install from a Developer Command Prompt or ensure MSVC is in your PATH
- Verify CUDA Toolkit version matches PyTorch (run `nvidia-smi` to check driver support)

### Detectron2 build fails on macOS
- Ensure Xcode Command Line Tools are installed: `xcode-select --install`
- Use `FORCE_CUDA=0` since macOS has no CUDA support
- If using Apple Silicon (M1/M2/M3), ensure you're using a native arm64 Python, not Rosetta

### CUDA out of memory during training
- Reduce batch size in the Train tab
- Use a smaller model (R50 instead of R101)
- Close other GPU-consuming applications

### pycocotools build fails on Windows
- Install via conda: `conda install -c conda-forge pycocotools`

### GUI not launching on Linux
- Install X11 dependencies: `sudo apt install libxcb-xinerama0 libxcb-cursor0`
- If running headless, you'll need a display server (X11 or Wayland)

### GUI not launching on macOS
- Verify PyQt6 is installed: `pip install PyQt6`
- On Apple Silicon, ensure you're not mixing arm64 and x86_64 packages

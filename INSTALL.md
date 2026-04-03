# Installation Guide

## Prerequisites

- **Python 3.10 or 3.11** (other versions are not supported due to Detectron2/PyTorch compatibility)
- **CUDA Toolkit 11.8 or 12.1** (must match your PyTorch build)
- **MSVC Build Tools 2019+** (required for compiling Detectron2 C++ extensions on Windows)

## Step-by-Step Setup

### 1. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/macOS
```

### 2. Install PyTorch

Install PyTorch matching your CUDA version. Visit https://pytorch.org/get-started/locally/ for the exact command.

Example for CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Example for CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

CPU only:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install pycocotools

On Windows, the standard `pycocotools` may fail to build. Use one of these options:

**Option A — pip (may require MSVC):**
```bash
pip install pycocotools
```

**Option B — Windows-specific package:**
```bash
pip install pycocotools-windows
```

**Option C — conda:**
```bash
conda install -c conda-forge pycocotools
```

### 4. Install Detectron2

Detectron2 does not provide prebuilt Windows wheels. Install from source:

```bash
pip install git+https://github.com/facebookresearch/detectron2.git
```

This requires MSVC Build Tools and a matching CUDA toolkit. If the build fails:
- Ensure `cl.exe` is in your PATH (run from a Developer Command Prompt)
- Verify your CUDA toolkit version matches your PyTorch installation
- Check that your Python version is 3.10 or 3.11

On Linux, the install is typically straightforward:
```bash
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

### 5. Install remaining dependencies

```bash
pip install PyQt6 pyqtgraph opencv-python Pillow numpy
```

Or install everything (except PyTorch and Detectron2) from requirements.txt:
```bash
pip install -r requirements.txt
```

### 6. Verify installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import detectron2; print('Detectron2:', detectron2.__version__)"
python -c "from PyQt6 import QtWidgets; print('PyQt6: OK')"
```

### 7. Run the application

```bash
python main.py
```

## Troubleshooting

### Detectron2 build fails on Windows
- Make sure you're using Python 3.10 or 3.11
- Install Visual Studio Build Tools with "Desktop development with C++"
- Run the install from a Developer Command Prompt or ensure MSVC is in your PATH

### CUDA out of memory during training
- Reduce batch size in the Train tab
- Use a smaller model (R50 instead of R101)
- Close other GPU-consuming applications

### pycocotools build fails
- Try `pip install pycocotools-windows` as an alternative
- Or install via conda: `conda install -c conda-forge pycocotools`

### GUI not launching
- Verify PyQt6 is installed: `pip install PyQt6`
- On Linux, you may need: `sudo apt install libxcb-xinerama0`

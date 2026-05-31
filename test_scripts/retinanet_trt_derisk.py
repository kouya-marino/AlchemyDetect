"""De-risk: can a RetinaNet ONNX be parsed by TensorRT (no RoiAlign / plugin walls)?

Exploratory/manual script (not a pytest test — lives outside tests/). Uses a
pretrained RetinaNet (no training needed — same op graph as a trained one). The
ONNX parse step is GPU-independent, so this works with whatever TensorRT is
installed, even if engine *build* would need a different version for your GPU.

Run (from a shell with CUDA/cuDNN/TensorRT on PATH):
    venv\\Scripts\\python.exe test_scripts\\retinanet_trt_derisk.py
"""

import os
import tempfile

import torch
from detectron2 import model_zoo
from detectron2.export import TracingAdapter
from detectron2.modeling import GeneralizedRCNN

CFG = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
out = os.path.join(tempfile.mkdtemp(prefix="retina_"), "retinanet.onnx")

print("Loading pretrained RetinaNet...")
model = model_zoo.get(CFG, trained=True)
model.eval()
print("meta-arch:", type(model).__name__, "| is GeneralizedRCNN:", isinstance(model, GeneralizedRCNN))

dev = "cuda" if torch.cuda.is_available() else "cpu"
model.to(dev)
sample = torch.rand(3, 800, 800, device=dev) * 255.0
adapter = TracingAdapter(model, [{"image": sample}])
adapter.eval()

print("Exporting ONNX (opset 17)...")
with torch.no_grad():
    torch.onnx.export(
        adapter,
        adapter.flattened_inputs,
        out,
        opset_version=17,
        input_names=["image"],
        dynamic_axes={"image": {1: "h", 2: "w"}},
    )
print("wrote", out)

# 1) Inspect ops
import onnx

m = onnx.load(out)
ops = sorted({n.op_type for n in m.graph.node})
print("\nONNX op types:", ops)
print("contains RoiAlign:", any("roi" in o.lower() for o in ops))
print("NMS-related ops:", [o for o in ops if "nms" in o.lower() or "NonMax" in o])

# 2) Try to parse with TensorRT
print("\n--- TensorRT parse test ---")
import tensorrt as trt

print("TensorRT", trt.__version__)
logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, "")  # register standard plugins (NMS etc.)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
with open(out, "rb") as f:
    ok = parser.parse(f.read())
if ok:
    print("PARSE OK -> RetinaNet graph is TensorRT-convertible")
else:
    print("PARSE FAILED:")
    for i in range(parser.num_errors):
        print("  ", parser.get_error(i))

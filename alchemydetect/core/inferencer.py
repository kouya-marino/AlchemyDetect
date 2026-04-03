"""Inference wrapper around Detectron2's DefaultPredictor."""

import cv2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


def load_predictor(config_yaml_path, weights_path, threshold=0.5):
    """Load a Detectron2 predictor from a saved config + weights pair.

    Args:
        config_yaml_path: Path to the saved config.yaml file.
        weights_path: Path to the .pth weights file.
        threshold: Confidence threshold for predictions.

    Returns:
        (DefaultPredictor, CfgNode)
    """
    cfg = get_cfg()
    cfg.merge_from_file(config_yaml_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    if hasattr(cfg.MODEL, "RETINANET"):
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
    cfg.freeze()

    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def run_inference_single(predictor, image_path):
    """Run inference on a single image.

    Args:
        predictor: A DefaultPredictor instance.
        image_path: Path to the image file.

    Returns:
        (original_image_bgr, instances) where instances is the prediction output.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    outputs = predictor(img)
    return img, outputs["instances"].to("cpu")


def visualize_predictions(image_bgr, instances, metadata=None):
    """Draw predictions on an image using Detectron2's Visualizer.

    Args:
        image_bgr: Original image in BGR format (numpy array).
        instances: Detectron2 Instances object (on CPU).
        metadata: Optional MetadataCatalog metadata for class names.

    Returns:
        Annotated image as RGB numpy array.
    """
    image_rgb = image_bgr[:, :, ::-1]
    if metadata is None:
        metadata = MetadataCatalog.get("__empty")

    v = Visualizer(image_rgb, metadata=metadata, scale=1.0, instance_mode=ColorMode.IMAGE)
    out = v.draw_instance_predictions(instances)
    return out.get_image()

"""Available Detectron2 model zoo entries."""

# Maps user-friendly name -> (model_zoo config path, task type)
MODEL_ZOO = {
    # Object Detection
    "Faster R-CNN (R50-FPN)": {
        "config": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        "task": "detection",
    },
    "Faster R-CNN (R101-FPN)": {
        "config": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
        "task": "detection",
    },
    "RetinaNet (R50-FPN)": {
        "config": "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
        "task": "detection",
    },
    "RetinaNet (R101-FPN)": {
        "config": "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
        "task": "detection",
    },
    # Instance Segmentation
    "Mask R-CNN (R50-FPN)": {
        "config": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "task": "instance_segmentation",
    },
    "Mask R-CNN (R101-FPN)": {
        "config": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        "task": "instance_segmentation",
    },
}


def get_model_names():
    """Return list of all available model names."""
    return list(MODEL_ZOO.keys())


def get_detection_models():
    """Return names of detection-only models."""
    return [k for k, v in MODEL_ZOO.items() if v["task"] == "detection"]


def get_segmentation_models():
    """Return names of instance segmentation models."""
    return [k for k, v in MODEL_ZOO.items() if v["task"] == "instance_segmentation"]


def get_config_path(model_name):
    """Return the model zoo config path for a given model name."""
    return MODEL_ZOO[model_name]["config"]


def get_task(model_name):
    """Return the task type ('detection' or 'instance_segmentation')."""
    return MODEL_ZOO[model_name]["task"]

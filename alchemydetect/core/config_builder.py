"""Build Detectron2 config from user-specified parameters."""

import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg

from .dataset_utils import get_num_classes, register_coco_dataset
from .model_catalog import get_config_path


def build_cfg(
    model_name,
    train_images_dir,
    train_json,
    output_dir,
    lr=0.0025,
    max_iter=1000,
    batch_size=2,
    val_images_dir=None,
    val_json=None,
    resume=False,
    weights_path=None,
):
    """Build a Detectron2 CfgNode from user selections.

    Args:
        model_name: Key from MODEL_ZOO (e.g. "Faster R-CNN (R50-FPN)")
        train_images_dir: Path to training images directory
        train_json: Path to COCO JSON annotations for training
        output_dir: Directory to save checkpoints and logs
        lr: Base learning rate
        max_iter: Maximum training iterations
        batch_size: Images per batch
        val_images_dir: Optional path to validation images directory
        val_json: Optional path to validation COCO JSON
        resume: Whether to resume from last checkpoint in output_dir
        weights_path: Optional path to custom weights (.pth). If None, uses model zoo pretrained.

    Returns:
        CfgNode ready for training.
    """
    config_path = get_config_path(model_name)
    num_classes = get_num_classes(train_json)

    # Register datasets
    train_dataset_name = "alchemy_train"
    register_coco_dataset(train_dataset_name, train_json, train_images_dir)

    val_dataset_name = None
    if val_json and val_images_dir:
        val_dataset_name = "alchemy_val"
        register_coco_dataset(val_dataset_name, val_json, val_images_dir)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_path))

    # Datasets
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,) if val_dataset_name else ()

    # Dataloader — use 0 workers to avoid nested multiprocessing issues on Windows
    cfg.DATALOADER.NUM_WORKERS = 0

    # Model weights
    if weights_path:
        cfg.MODEL.WEIGHTS = weights_path
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)

    # Solver
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = []  # No LR decay for simplicity
    cfg.SOLVER.CHECKPOINT_PERIOD = max(max_iter // 5, 100)

    # Number of classes
    if hasattr(cfg.MODEL, "ROI_HEADS"):
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    if hasattr(cfg.MODEL, "RETINANET"):
        cfg.MODEL.RETINANET.NUM_CLASSES = num_classes

    # Device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Output
    cfg.OUTPUT_DIR = output_dir

    cfg.freeze()
    return cfg

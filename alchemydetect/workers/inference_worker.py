"""Background thread for running Detectron2 (.pth) inference."""

from alchemydetect.workers.inference_base import ImageInferenceWorker


class InferenceWorker(ImageInferenceWorker):
    """Runs inference on images using a Detectron2 DefaultPredictor."""

    INIT_ERROR_LABEL = "Inference initialization error"

    def __init__(self, config_yaml_path, weights_path, image_paths, threshold=0.5, class_names=None):
        """
        Args:
            config_yaml_path: Path to saved config.yaml.
            weights_path: Path to .pth weights file.
            image_paths: List of image file paths.
            threshold: Confidence threshold.
            class_names: Optional list of class name strings.
        """
        super().__init__(image_paths, threshold, class_names)
        self._config_yaml_path = config_yaml_path
        self._weights_path = weights_path

    def _setup(self):
        from alchemydetect.core.inferencer import load_predictor

        predictor, _ = load_predictor(self._config_yaml_path, self._weights_path, self._threshold)
        metadata = self._make_metadata("__alchemy_inference")

        def detect(image_bgr):
            return predictor(image_bgr)["instances"].to("cpu")

        return detect, metadata

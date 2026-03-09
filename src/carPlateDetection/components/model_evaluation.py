import json
from pathlib import Path
from carPlateDetection import logger
from carPlateDetection.entity.config_entity import ModelEvaluationConfig
from carPlateDetection.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self):
        """Run YOLO validation and save metrics to a JSON file.

        Metrics saved include:
            - mAP50, mAP50-95
            - Precision, Recall
            - Fitness score
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required. Install it with: "
                "pip install ultralytics"
            )

        logger.info(f"Loading model from: {self.config.model_path}")
        model = YOLO(str(self.config.model_path))

        logger.info("Running validation …")
        metrics = model.val(
            data=str(self.config.data_yaml),
            imgsz=self.config.image_size,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            verbose=True,
        )

        # Extract scalar values from the metrics object
        results_dict = {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "fitness": float(metrics.fitness),
        }

        logger.info(f"Evaluation results: {results_dict}")

        save_json(
            path=Path(self.config.metric_file_name),
            data=results_dict,
        )

        logger.info(
            f"Metrics saved to: {self.config.metric_file_name}"
        )

        return results_dict

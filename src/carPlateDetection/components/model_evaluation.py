from pathlib import Path
from carPlateDetection import logger
from carPlateDetection.utils.common import save_json
from carPlateDetection.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self):
        """Run YOLO validation and save metrics to JSON."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Run: pip install ultralytics")

        model_path   = Path(self.config.model_path).resolve()
        data_yaml    = Path(self.config.data_yaml).resolve()
        metric_file  = Path(self.config.metric_file_name)
        metric_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading model : {model_path}")
        logger.info(f"Data yaml     : {data_yaml}")

        model   = YOLO(str(model_path))
        metrics = model.val(
            data=str(data_yaml),
            imgsz=self.config.image_size,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            verbose=True,
        )

        results = {
            "mAP50":     float(metrics.box.map50),
            "mAP50_95":  float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall":    float(metrics.box.mr),
            "fitness":   float(metrics.fitness),
        }

        logger.info(f"Evaluation results: {results}")
        save_json(path=metric_file, data=results)
        logger.info(f"Metrics saved to: {metric_file}")
        return results
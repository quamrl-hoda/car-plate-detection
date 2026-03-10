from pathlib import Path
from carPlateDetection import logger
from carPlateDetection.utils.common import save_json
from carPlateDetection.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self) -> dict:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Run: pip install ultralytics")

        model_path  = Path(self.config.model_path).resolve()
        data_yaml   = Path(self.config.data_yaml).resolve()
        metric_file = Path(self.config.metric_file_name)
        metric_file.parent.mkdir(parents=True, exist_ok=True)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Evaluating {model_path}")
        model   = YOLO(str(model_path))
        metrics = model.val(
            data    = str(data_yaml),
            imgsz   = self.config.image_size,
            conf    = self.config.conf_threshold,
            iou     = self.config.iou_threshold,
            device  = "cpu",
            verbose = True,
        )

        results = {
            "mAP50":     round(float(metrics.box.map50), 4),
            "mAP50_95":  round(float(metrics.box.map),   4),
            "precision": round(float(metrics.box.mp),    4),
            "recall":    round(float(metrics.box.mr),    4),
            "fitness":   round(float(metrics.fitness),   4),
        }
        logger.info(f"Metrics: {results}")
        save_json(path=metric_file, data=results)
        return results

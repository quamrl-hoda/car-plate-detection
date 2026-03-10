"""
model_trainer.py
================
Trains YOLOv8n — mirrors the Streamlit app's model exactly.

Streamlit app loads:
  model = YOLO('/Users/.../best.pt')

This trainer produces that best.pt at:
  artifacts/model_trainer/car_plate_detector/weights/best.pt
"""
from pathlib import Path
from carPlateDetection import logger
from carPlateDetection.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("pip install ultralytics")

        project_abs = Path(self.config.project).resolve()
        data_yaml   = Path(self.config.data_yaml).resolve()
        project_abs.mkdir(parents=True, exist_ok=True)

        if not data_yaml.exists():
            raise FileNotFoundError(
                f"data.yaml not found: {data_yaml}\n"
                "Run Stage 3 (data_transformation) first."
            )

        logger.info(
            f"Training  model={self.config.model_name}  "
            f"epochs={self.config.epochs}  imgsz={self.config.image_size}  "
            f"batch={self.config.batch_size}  project={project_abs}"
        )

        model = YOLO(self.config.model_name)   # yolov8n.pt pretrained
        model.train(
            data     = str(data_yaml),
            epochs   = self.config.epochs,
            imgsz    = self.config.image_size,
            batch    = self.config.batch_size,
            device   = "cpu",       # same as Streamlit app
            project  = str(project_abs),
            name     = self.config.name,
            exist_ok = True,
            cache    = False,
            workers  = 0,           # 0 avoids Windows fork errors
            verbose  = True,
        )

        best = project_abs / self.config.name / "weights" / "best.pt"
        if not best.exists():
            raise FileNotFoundError(
                f"Training finished but best.pt missing: {best}\n"
                "Check YOLO logs above for errors."
            )
        logger.info(f"best.pt saved: {best}")

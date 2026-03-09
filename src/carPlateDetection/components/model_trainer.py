import os
from pathlib import Path
from carPlateDetection import logger
from carPlateDetection.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        """Train YOLOv8 for car plate detection.

        YOLO resolves the `project` argument relative to its own working
        directory and also prepends `runs/detect/` unless you pass an
        absolute path. We resolve to absolute here to guarantee weights
        land exactly at:
            artifacts/model_trainer/car_plate_detector/weights/best.pt
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Run: pip install ultralytics")

        # ── Resolve all paths to absolute ─────────────────────────────
        project_abs  = Path(self.config.project).resolve()
        data_yaml_abs = Path(self.config.data_yaml).resolve()

        os.makedirs(project_abs, exist_ok=True)

        logger.info(f"Loading base model : {self.config.model_name}")
        logger.info(f"Data yaml          : {data_yaml_abs}")
        logger.info(f"Project dir        : {project_abs}")
        logger.info(f"Run name           : {self.config.name}")

        model = YOLO(self.config.model_name)

        logger.info(
            f"Starting training — epochs={self.config.epochs}, "
            f"imgsz={self.config.image_size}, batch={self.config.batch_size}"
        )

        model.train(
            data=str(data_yaml_abs),      # absolute path → no ambiguity
            epochs=self.config.epochs,
            imgsz=self.config.image_size,
            batch=self.config.batch_size,
            project=str(project_abs),     # absolute path → YOLO won't prepend runs/detect/
            name=self.config.name,
            exist_ok=True,
            verbose=True,
        )

        # ── Verify weights were saved where DVC expects them ──────────
        expected = project_abs / self.config.name / "weights" / "best.pt"
        if not expected.exists():
            raise FileNotFoundError(
                f"Training finished but best.pt not found at: {expected}\n"
                f"Check the YOLO project/name settings."
            )

        logger.info("Training complete.")
        logger.info(f"Best weights saved to: {expected}")
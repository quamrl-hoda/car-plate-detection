import os
from pathlib import Path
from carPlateDetection import logger
from carPlateDetection.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        """Train a YOLOv8 model for car plate detection.

        The trained weights are saved to:
            <project>/<name>/weights/best.pt
            <project>/<name>/weights/last.pt
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required. Install it with: "
                "pip install ultralytics"
            )

        logger.info(
            f"Loading base model: {self.config.model_name}"
        )
        model = YOLO(self.config.model_name)

        logger.info(
            f"Starting training — epochs={self.config.epochs}, "
            f"imgsz={self.config.image_size}, "
            f"batch={self.config.batch_size}"
        )

        results = model.train(
            data=str(self.config.data_yaml),
            epochs=self.config.epochs,
            imgsz=self.config.image_size,
            batch=self.config.batch_size,
            project=str(self.config.project),
            name=self.config.name,
            exist_ok=True,   # overwrite previous run if re-training
            verbose=True,
        )

        logger.info("Training complete.")
        logger.info(
            f"Best weights saved to: "
            f"{self.config.project}/{self.config.name}/weights/best.pt"
        )

        return results

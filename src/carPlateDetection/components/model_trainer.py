"""
model_trainer.py
================
Fine-tunes YOLOv8n on the license-plate dataset.

Key production decisions:
  - project path is ALWAYS resolved to absolute before passing to YOLO.
    Without this YOLO prepends `runs/detect/` giving you:
    runs/detect/artifacts/model_trainer/car_plate_detector/weights/best.pt
  - exist_ok=True so DVC --force reruns don't crash on existing output dir.
  - device auto-selects GPU if available, falls back to CPU.
  - All hyperparameters come from ModelTrainerConfig (params.yaml via DVC).
"""
from pathlib import Path

from carPlateDetection import logger
from carPlateDetection.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

        # Resolve absolute paths — critical to prevent YOLO path-doubling
        project_abs = Path(self.config.project).resolve()
        data_yaml   = Path(self.config.data_yaml).resolve()
        project_abs.mkdir(parents=True, exist_ok=True)

        if not data_yaml.exists():
            raise FileNotFoundError(
                f"data.yaml not found at {data_yaml}.\n"
                "Run data_transformation stage first."
            )

        logger.info("=" * 55)
        logger.info("  MODEL TRAINER")
        logger.info(f"  Model    : {self.config.model_name}")
        logger.info(f"  Data     : {data_yaml}")
        logger.info(f"  Project  : {project_abs}")
        logger.info(f"  Run name : {self.config.name}")
        logger.info(f"  Epochs   : {self.config.epochs}")
        logger.info(f"  Img size : {self.config.image_size}")
        logger.info(f"  Batch    : {self.config.batch_size}")
        logger.info("=" * 55)

        model = YOLO(self.config.model_name)
        model.train(
            data     = str(data_yaml),
            epochs   = self.config.epochs,
            imgsz    = self.config.image_size,
            batch    = self.config.batch_size,
            project  = str(project_abs),   # MUST be absolute
            name     = self.config.name,
            exist_ok = True,               # Safe for DVC reruns
            cache    = False,
            verbose  = True,
        )

        best = project_abs / self.config.name / "weights" / "best.pt"
        if not best.exists():
            raise FileNotFoundError(
                f"Training finished but best.pt not found at: {best}\n"
                "Training may have failed silently — check YOLO logs above."
            )
        logger.info(f"Training complete. Best weights saved: {best}")

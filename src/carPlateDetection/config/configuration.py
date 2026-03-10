from pathlib import Path
from carPlateDetection.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from carPlateDetection.utils.common import read_yaml, create_directories
from carPlateDetection.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config.data_ingestion
        create_directories([cfg.root_dir])
        return DataIngestionConfig(
            root_dir=Path(cfg.root_dir),
            source_URL=cfg.source_URL,
            local_data_file=Path(cfg.local_data_file),
            unzip_dir=Path(cfg.unzip_dir),
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        cfg = self.config.data_validation
        create_directories([cfg.root_dir])
        return DataValidationConfig(
            root_dir=Path(cfg.root_dir),
            data_dir=Path(cfg.data_dir),
            status_file=cfg.status_file,
            required_files=cfg.required_files,
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        cfg = self.config.data_transformation
        create_directories([cfg.root_dir])
        return DataTransformationConfig(
            root_dir=Path(cfg.root_dir),
            data_dir=Path(cfg.data_dir),
            image_size=cfg.image_size,
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        cfg    = self.config.model_trainer
        params = self.params.TrainingArguments
        create_directories([cfg.root_dir])
        return ModelTrainerConfig(
            root_dir=Path(cfg.root_dir),
            data_yaml=Path(cfg.data_yaml),
            model_name=cfg.model_name,
            epochs=params.epochs,
            image_size=params.image_size,
            batch_size=params.batch_size,
            project=Path(cfg.project),
            name=cfg.name,
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        cfg    = self.config.model_evaluation
        params = self.params.TrainingArguments
        create_directories([cfg.root_dir])
        return ModelEvaluationConfig(
            root_dir=Path(cfg.root_dir),
            model_path=Path(cfg.model_path),
            data_yaml=Path(cfg.data_yaml),
            metric_file_name=Path(cfg.metric_file_name),
            image_size=params.image_size,
            conf_threshold=params.conf_threshold,
            iou_threshold=params.iou_threshold,
        )

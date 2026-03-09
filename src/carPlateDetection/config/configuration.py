from carPlateDetection.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from carPlateDetection.utils.common import read_yaml, create_directories
from carPlateDetection.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)
from pathlib import Path


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    # ------------------------------------------------------------------ #
    #  Data Ingestion
    # ------------------------------------------------------------------ #
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
        )

    # ------------------------------------------------------------------ #
    #  Data Validation
    # ------------------------------------------------------------------ #
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_directories([config.root_dir])

        return DataValidationConfig(
            root_dir=Path(config.root_dir),
            data_dir=Path(config.data_dir),
            status_file=config.status_file,
            required_files=config.required_files,
        )

    # ------------------------------------------------------------------ #
    #  Data Transformation
    # ------------------------------------------------------------------ #
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])

        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_dir=Path(config.data_dir),
            image_size=config.image_size,
        )

    # ------------------------------------------------------------------ #
    #  Model Trainer
    # ------------------------------------------------------------------ #
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments
        create_directories([config.root_dir])

        return ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            data_yaml=Path(config.data_yaml),
            model_name=config.model_name,
            epochs=params.epochs,
            image_size=params.image_size,
            batch_size=params.batch_size,
            project=Path(config.project),
            name=config.name,
        )

    # ------------------------------------------------------------------ #
    #  Model Evaluation
    # ------------------------------------------------------------------ #
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.TrainingArguments
        create_directories([config.root_dir])

        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            data_yaml=Path(config.data_yaml),
            metric_file_name=Path(config.metric_file_name),
            image_size=params.image_size,
            conf_threshold=params.conf_threshold,
            iou_threshold=params.iou_threshold,
        )

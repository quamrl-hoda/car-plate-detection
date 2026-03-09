from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    data_dir: Path
    status_file: str
    required_files: list


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_dir: Path
    image_size: int


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_yaml: Path
    model_name: str
    epochs: int
    image_size: int
    batch_size: int
    project: Path
    name: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    data_yaml: Path
    metric_file_name: Path
    image_size: int
    conf_threshold: float
    iou_threshold: float

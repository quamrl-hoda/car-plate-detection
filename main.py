from carPlateDetection import logger

from carPlateDetection.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from carPlateDetection.pipeline.stage_02_data_validation import (
    DataValidationTrainingPipeline,
)
from carPlateDetection.pipeline.stage_03_data_transformation import (
    DataTransformationTrainingPipeline,
)
from carPlateDetection.pipeline.stage_04_model_trainer import (
    ModelTrainerTrainingPipeline,
)
from carPlateDetection.pipeline.stage_05_model_evaluation import (
    ModelEvaluationTrainingPipeline,
)


# ------------------------------------------------------------------ #
#  Stage 1 – Data Ingestion
# ------------------------------------------------------------------ #
STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# ------------------------------------------------------------------ #
#  Stage 2 – Data Validation
# ------------------------------------------------------------------ #
STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataValidationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# ------------------------------------------------------------------ #
#  Stage 3 – Data Transformation
# ------------------------------------------------------------------ #
STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataTransformationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# ------------------------------------------------------------------ #
#  Stage 4 – Model Trainer
# ------------------------------------------------------------------ #
STAGE_NAME = "Model Trainer Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainerTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# ------------------------------------------------------------------ #
#  Stage 5 – Model Evaluation
# ------------------------------------------------------------------ #
STAGE_NAME = "Model Evaluation Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelEvaluationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

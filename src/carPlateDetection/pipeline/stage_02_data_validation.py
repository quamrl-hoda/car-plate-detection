from carPlateDetection import logger
from carPlateDetection.config.configuration import ConfigurationManager
from carPlateDetection.components.data_validation import DataValidation

STAGE_NAME = "Data Validation"

class DataValidationTrainingPipeline:
    def main(self):
        cfg = ConfigurationManager()
        dv  = DataValidation(config=cfg.get_data_validation_config())
        dv.validate_all_files_exist()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        DataValidationTrainingPipeline().main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e); raise e

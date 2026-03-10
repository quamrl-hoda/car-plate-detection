from carPlateDetection import logger
from carPlateDetection.config.configuration import ConfigurationManager
from carPlateDetection.components.data_transformation import DataTransformation

STAGE_NAME = "Data Transformation"

class DataTransformationTrainingPipeline:
    def main(self):
        cfg = ConfigurationManager()
        dt  = DataTransformation(config=cfg.get_data_transformation_config())
        dt.process_images()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        DataTransformationTrainingPipeline().main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e); raise e

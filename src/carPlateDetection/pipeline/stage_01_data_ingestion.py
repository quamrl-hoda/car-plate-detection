from carPlateDetection import logger
from carPlateDetection.config.configuration import ConfigurationManager
from carPlateDetection.components.data_ingestion import DataIngestion

STAGE_NAME = "Data Ingestion"

class DataIngestionTrainingPipeline:
    def main(self):
        cfg = ConfigurationManager()
        di  = DataIngestion(config=cfg.get_data_ingestion_config())
        di.download_file()
        di.extract_zip_file()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        DataIngestionTrainingPipeline().main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e); raise e

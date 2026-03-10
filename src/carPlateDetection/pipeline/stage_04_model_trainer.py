from carPlateDetection import logger
from carPlateDetection.config.configuration import ConfigurationManager
from carPlateDetection.components.model_trainer import ModelTrainer

STAGE_NAME = "Model Trainer"

class ModelTrainerTrainingPipeline:
    def main(self):
        cfg = ConfigurationManager()
        mt  = ModelTrainer(config=cfg.get_model_trainer_config())
        mt.train()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        ModelTrainerTrainingPipeline().main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e); raise e

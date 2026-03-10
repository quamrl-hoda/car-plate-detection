from carPlateDetection import logger
from carPlateDetection.config.configuration import ConfigurationManager
from carPlateDetection.components.model_evaluation import ModelEvaluation

STAGE_NAME = "Model Evaluation"

class ModelEvaluationTrainingPipeline:
    def main(self):
        cfg = ConfigurationManager()
        me  = ModelEvaluation(config=cfg.get_model_evaluation_config())
        me.evaluate()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        ModelEvaluationTrainingPipeline().main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e); raise e

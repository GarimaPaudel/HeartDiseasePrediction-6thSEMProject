import numpy as np

from src.components.model_evaluator import ModelEvaluator
from src.utils.common import read_yaml, save_json
from src.utils.logger import logger

STAGE = "Model Evaluation"


def main(model=None, X_test: np.ndarray = None, y_test: np.ndarray = None):
    config = read_yaml("config/config.yaml")
    evaluator = ModelEvaluator(config.evaluation)
    metrics = evaluator.evaluate(model, X_test, y_test)

    report_path = f"{config.evaluation.report_dir}/evaluation_report.json"
    save_json(report_path, metrics)
    return metrics


if __name__ == "__main__":
    logger.info(f">>>>>> Stage: {STAGE} started <<<<<<")
    logger.warning("This stage requires a trained model — run via run_pipeline.py")
    logger.info(f">>>>>> Stage: {STAGE} completed <<<<<<\n")

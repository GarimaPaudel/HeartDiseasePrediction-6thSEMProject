import numpy as np

from src.components.model_evaluator import ModelEvaluator
from src.utils.common import load_json, read_yaml, save_json
from src.utils.logger import logger

STAGE = "Model Evaluation"


def main(model=None, X_test: np.ndarray = None, y_test: np.ndarray = None,
         model_name: str = None, best_params: dict = None):
    config = read_yaml("config/config.yaml")
    evaluator = ModelEvaluator(config.evaluation)
    metrics = evaluator.evaluate(model, X_test, y_test, model_name, best_params)

    report_path = f"{config.evaluation.report_dir}/evaluation_report.json"

    try:
        existing = load_json(report_path)
        if existing.get("objective_value", 0) >= metrics["objective_value"]:
            logger.info(
                f"Existing report (obj={existing['objective_value']}) is better than "
                f"current (obj={metrics['objective_value']}). Report not updated."
            )
            return metrics
    except FileNotFoundError:
        pass

    save_json(report_path, metrics)
    logger.info(f"Best report updated → {metrics['model']} (obj={metrics['objective_value']})")
    return metrics


if __name__ == "__main__":
    logger.info(f">>>>>> Stage: {STAGE} started <<<<<<")
    logger.warning("This stage requires a trained model — run via run_pipeline.py")
    logger.info(f">>>>>> Stage: {STAGE} completed <<<<<<\n")

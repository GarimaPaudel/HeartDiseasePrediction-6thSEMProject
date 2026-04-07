import numpy as np
from box import ConfigBox
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.utils.logger import logger


class ModelEvaluator:
    def __init__(self, config: ConfigBox):
        self.sensitivity_weight = config.sensitivity_weight

    def evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        accuracy = accuracy_score(y_test, y_pred)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        objective = accuracy + sensitivity * self.sensitivity_weight

        report = classification_report(y_test, y_pred, output_dict=True)

        metrics = {
            "accuracy": round(float(accuracy), 6),
            "sensitivity": round(float(sensitivity), 6),
            "specificity": round(float(specificity), 6),
            "objective_value": round(float(objective), 6),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

        logger.info(
            f"Evaluation — accuracy={accuracy:.4f} | "
            f"sensitivity={sensitivity:.4f} | objective={objective:.4f}"
        )
        return metrics

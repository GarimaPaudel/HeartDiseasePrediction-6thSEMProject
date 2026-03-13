from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import logger


class FeatureEngineer:
    """
    Canonical BMI computation shared by both the training pipeline and app.py.
    Height is always expected in centimetres (cm).
    """

    # BMI class boundaries matching the notebook exactly
    _BMI_CONDITIONS = [
        lambda b: b <= 15,
        lambda b: (b > 15) & (b <= 18.5),
        lambda b: (b > 18.5) & (b <= 25),
        lambda b: (b > 25) & (b <= 30),
        lambda b: (b > 30) & (b <= 35),
        lambda b: (b > 35) & (b <= 40),
    ]
    _BMI_CLASSES = [0, 1, 2, 3, 4, 5]
    _BMI_DEFAULT = 6  # bmi > 40

    def compute_bmi(self, weight_kg: float, height_cm: float) -> float:
        """BMI = weight(kg) / height(m)^2  — height_cm converted internally."""
        return weight_kg * 10_000 / (height_cm ** 2)

    def compute_bmi_class(self, bmi: float) -> int:
        for i, condition in enumerate(self._BMI_CONDITIONS):
            if condition(np.array([bmi]))[0]:
                return self._BMI_CLASSES[i]
        return self._BMI_DEFAULT

    def run(self, processed_data_path: Path) -> Path:
        df = pd.read_csv(processed_data_path)
        logger.info(f"Feature engineering input: shape={df.shape}")

        bmi_values = df["weight"] * 10_000 / (df["height"] ** 2)
        df["bmi"] = bmi_values.round(4)

        conditions = [
            bmi_values <= 15,
            (bmi_values > 15) & (bmi_values <= 18.5),
            (bmi_values > 18.5) & (bmi_values <= 25),
            (bmi_values > 25) & (bmi_values <= 30),
            (bmi_values > 30) & (bmi_values <= 35),
            (bmi_values > 35) & (bmi_values <= 40),
        ]
        df["bmi_class"] = np.select(conditions, self._BMI_CLASSES, default=self._BMI_DEFAULT)

        # Overwrite in-place (feature engineering adds columns to the same file)
        df.to_csv(processed_data_path, index=False)
        logger.info(
            f"Feature engineering done: added bmi, bmi_class | shape={df.shape}"
        )
        return processed_data_path

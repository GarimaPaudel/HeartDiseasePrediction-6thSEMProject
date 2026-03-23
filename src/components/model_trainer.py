from pathlib import Path

import pandas as pd
from box import ConfigBox
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.logger import logger

# Feature columns the model expects — order matters for app.py compatibility
FEATURE_COLS = [
    "age", "male", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active", "bmi", "bmi_class",
]
TARGET_COL = "cardio"

_MODEL_MAP = {
    "RandomForest": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    "DecisionTree": DecisionTreeClassifier,
    "KNN": KNeighborsClassifier,
}

# Models that require feature scaling
_NEEDS_SCALING = {"LogisticRegression", "KNN"}


class ModelTrainer:
    def __init__(self, config: ConfigBox, params: ConfigBox):
        self.config = config
        self.params = params

    def _build_estimator(self, model_name: str):
        estimator_cls = _MODEL_MAP[model_name]
        if model_name in _NEEDS_SCALING:
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", estimator_cls()),
            ])
        return estimator_cls()

    def _build_param_grid(self, model_name: str) -> dict:
        raw = dict(self.params[model_name])
        if model_name in _NEEDS_SCALING:
            # Prefix params with 'clf__' for Pipeline compatibility
            return {f"clf__{k}": v for k, v in raw.items()}
        return raw

    def run(self, data_path: Path):
        df = pd.read_csv(data_path)
        X = df[FEATURE_COLS].values
        y = df[TARGET_COL].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            stratify=y if self.config.stratify else None,
            random_state=self.config.random_state,
        )
        logger.info(
            f"Train/test split: train={len(X_train)}, test={len(X_test)}"
        )

        model_name = self.params.active_model
        estimator = self._build_estimator(model_name)
        param_grid = self._build_param_grid(model_name)

        cv = StratifiedKFold(
            n_splits=self.config.kfold_splits,
            shuffle=True,
            random_state=self.config.random_state,
        )
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )

        logger.info(f"Starting GridSearchCV for {model_name} ...")
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        logger.info(f"Best params: {best_params}")
        logger.info(f"Best CV accuracy: {grid_search.best_score_:.4f}")

        return best_model, best_params, X_test, y_test

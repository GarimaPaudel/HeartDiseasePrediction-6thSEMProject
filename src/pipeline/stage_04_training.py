from pathlib import Path

from src.components.model_trainer import ModelTrainer
from src.utils.common import read_yaml
from src.utils.logger import logger

STAGE = "Model Training"


def main(featured_path: Path = None):
    config = read_yaml("config/config.yaml")
    params = read_yaml("config/params.yaml")
    if featured_path is None:
        featured_path = (
            Path(config.data_preprocessing.processed_data_dir)
            / config.data_preprocessing.processed_file_name
        )
    trainer = ModelTrainer(config.training, params)
    model, best_params, X_test, y_test = trainer.run(featured_path)
    return model, best_params, X_test, y_test


if __name__ == "__main__":
    logger.info(f">>>>>> Stage: {STAGE} started <<<<<<")
    main()
    logger.info(f">>>>>> Stage: {STAGE} completed <<<<<<\n")

from pathlib import Path

from src.components.feature_engineering import FeatureEngineer
from src.utils.common import read_yaml
from src.utils.logger import logger

STAGE = "Feature Engineering"


def main(processed_path: Path = None):
    config = read_yaml("config/config.yaml")
    if processed_path is None:
        processed_path = (
            Path(config.data_preprocessing.processed_data_dir)
            / config.data_preprocessing.processed_file_name
        )
    fe = FeatureEngineer()
    featured_path = fe.run(processed_path)
    return featured_path


if __name__ == "__main__":
    logger.info(f">>>>>> Stage: {STAGE} started <<<<<<")
    main()
    logger.info(f">>>>>> Stage: {STAGE} completed <<<<<<\n")

from pathlib import Path

from src.components.data_preprocessing import DataPreprocessor
from src.utils.common import read_yaml
from src.utils.logger import logger

STAGE = "Data Preprocessing"


def main(raw_path: Path = None):
    config = read_yaml("config/config.yaml")
    if raw_path is None:
        raw_path = (
            Path(config.data_ingestion.raw_data_dir)
            / config.data_ingestion.raw_file_name
        )
    preprocessor = DataPreprocessor(config.data_preprocessing)
    processed_path = preprocessor.run(raw_path)
    return processed_path


if __name__ == "__main__":
    logger.info(f">>>>>> Stage: {STAGE} started <<<<<<")
    main()
    logger.info(f">>>>>> Stage: {STAGE} completed <<<<<<\n")

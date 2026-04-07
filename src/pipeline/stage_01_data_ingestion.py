from src.components.data_ingestion import DataIngestion
from src.utils.common import read_yaml
from src.utils.logger import logger

STAGE = "Data Ingestion"


def main():
    config = read_yaml("config/config.yaml")
    ingestion = DataIngestion(config.data_ingestion)
    raw_path = ingestion.run()
    return raw_path


if __name__ == "__main__":
    logger.info(f">>>>>> Stage: {STAGE} started <<<<<<")
    main()
    logger.info(f">>>>>> Stage: {STAGE} completed <<<<<<\n")

from src.components.model_registry import ModelRegistry
from src.utils.common import read_yaml
from src.utils.logger import logger

STAGE = "Model Registration"


def main(model=None, metrics: dict = None, params: dict = None) -> str:
    config = read_yaml("config/config.yaml")
    registry = ModelRegistry(config.model_registry)
    version = registry.register(model, metrics, params)
    return version


if __name__ == "__main__":
    logger.info(f">>>>>> Stage: {STAGE} started <<<<<<")
    logger.warning("This stage requires a trained model — run via run_pipeline.py")
    logger.info(f">>>>>> Stage: {STAGE} completed <<<<<<\n")

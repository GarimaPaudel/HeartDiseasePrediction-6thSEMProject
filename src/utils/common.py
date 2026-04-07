import os
import json
import pickle
from pathlib import Path
from typing import Any

import yaml
from box import ConfigBox

from src.utils.logger import logger


def read_yaml(path: str) -> ConfigBox:
    """Load a YAML file and return as a ConfigBox (dot-access dict)."""
    with open(path, "r") as f:
        content = yaml.safe_load(f)
    logger.info(f"Loaded YAML: {path}")
    return ConfigBox(content)


def save_json(path: str | Path, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved JSON: {path}")


def load_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_pickle(path: str | Path, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved pickle: {path}")


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def create_directories(paths: list) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)
        logger.debug(f"Created directory: {p}")

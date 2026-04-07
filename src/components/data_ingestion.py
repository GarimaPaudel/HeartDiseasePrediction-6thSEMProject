import shutil
from pathlib import Path

import pandas as pd
from box import ConfigBox
from src.utils.common import create_directories
from src.utils.logger import logger


class DataIngestion:
    def __init__(self, config: ConfigBox):
        self.config = config

    def run(self) -> Path:
        create_directories([self.config.raw_data_dir])

        dest = Path(self.config.raw_data_dir) / self.config.raw_file_name
        shutil.copy(self.config.source_path, dest)

        df = pd.read_csv(dest)
        logger.info(
            f"Data ingested: {dest} | shape={df.shape} | columns={list(df.columns)}"
        )
        return dest

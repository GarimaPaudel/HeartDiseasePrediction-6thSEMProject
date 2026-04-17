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
        src = Path(self.config.source_path)
        if src.resolve() != dest.resolve():
            shutil.copy(src, dest)

        df = pd.read_csv(dest)
        logger.info(
            f"Data ingested: {dest} | shape={df.shape} | columns={list(df.columns)}"
        )
        return dest

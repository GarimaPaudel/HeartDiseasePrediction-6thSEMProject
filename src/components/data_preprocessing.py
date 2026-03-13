from pathlib import Path

import pandas as pd
from box import ConfigBox

from src.utils.common import create_directories
from src.utils.logger import logger


class DataPreprocessor:
    def __init__(self, config: ConfigBox):
        self.config = config

    def _convert_age(self, df: pd.DataFrame) -> pd.DataFrame:
        # Age is stored in days in the raw dataset — convert to years
        df["age"] = (df["age"] / 365).round().astype(int)
        return df

    def _encode_gender(self, df: pd.DataFrame) -> pd.DataFrame:
        # Raw: 1=female, 2=male → encode as binary 0/1, rename to 'male'
        df["gender"] = df["gender"].map({1: 0, 2: 1})
        df.rename(columns={"gender": "male"}, inplace=True)
        return df

    def _filter_height(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df[
            (df["height"] >= self.config.height_min_cm)
            & (df["height"] <= self.config.height_max_cm)
        ]
        logger.info(f"Height filter removed {before - len(df)} rows")
        return df

    def _filter_blood_pressure(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df[
            (df["ap_hi"] >= self.config.ap_hi_min)
            & (df["ap_hi"] <= self.config.ap_hi_max)
            & (df["ap_lo"] >= self.config.ap_lo_min)
            & (df["ap_lo"] <= self.config.ap_lo_max)
            & (df["ap_hi"] >= df["ap_lo"])  # systolic must be >= diastolic
        ]
        logger.info(f"Blood pressure filter removed {before - len(df)} rows")
        return df

    def run(self, raw_data_path: Path) -> Path:
        create_directories([self.config.processed_data_dir])

        df = pd.read_csv(raw_data_path)
        logger.info(f"Raw data loaded: shape={df.shape}")

        df = self._convert_age(df)
        df = self._encode_gender(df)
        df = self._filter_height(df)
        df = self._filter_blood_pressure(df)

        # Drop id column if present
        if "id" in df.columns:
            df.drop(columns=["id"], inplace=True)

        df.reset_index(drop=True, inplace=True)

        out_path = Path(self.config.processed_data_dir) / self.config.processed_file_name
        df.to_csv(out_path, index=False)
        logger.info(f"Preprocessed data saved: {out_path} | shape={df.shape}")
        return out_path

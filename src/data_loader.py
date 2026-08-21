"""
Data loading utilities.
Reads raw CSV data, normalizes column names, ensures date types.
"""
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.common.config import settings
from src.common.logging_config import get_logger

logger = get_logger(__name__)


REQUIRED_COLUMNS = [
    "date", "site_id", "consumed_tonnes", "planned_pour_tonnes",
    "rain_mm", "avg_temp_c", "silo_capacity", "behavior",
    "cement_type", "region",
]


def load_raw(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load raw CSV data."""
    path = Path(path) if path else settings.DATA_RAW
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    return _normalize(df)


def load_processed(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load already-processed CSV."""
    path = Path(path) if path else settings.DATA_PROCESSED
    logger.info("Loading processed data from %s", path)
    df = pd.read_csv(path)
    return _normalize(df)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and dtypes."""
    # Common rename patterns
    if "ite_id" in df.columns and "site_id" not in df.columns:
        df = df.rename(columns={"ite_id": "site_id"})

    # Date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Sort
    if "site_id" in df.columns and "date" in df.columns:
        df = df.sort_values(["site_id", "date"]).reset_index(drop=True)

    # Validate
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("Missing expected columns: %s", missing)

    return df


def load_or_generate_demo() -> pd.DataFrame:
    """Load processed data, or fall back to raw if processed doesn't exist."""
    processed = settings.DATA_PROCESSED / "operations_cleaned.csv"
    if processed.exists():
        return load_processed(processed)
    logger.info("Processed data not found, loading raw.")
    return load_raw()

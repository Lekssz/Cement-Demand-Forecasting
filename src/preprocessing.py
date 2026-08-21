"""
Preprocessing pipeline for the cement demand forecasting model.

Aggregates daily data to weekly, engineers lag/rolling features,
and produces multi-step targets.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.common.config import settings
from src.common.logging_config import get_logger

logger = get_logger(__name__)

LAG_COLS = [1, 2, 4, 8]
ROLLING_WINDOWS = [4, 8]

NUMERIC_FEATURES = [
    "consumed_tonnes_lag_1", "consumed_tonnes_lag_2",
    "consumed_tonnes_lag_4", "consumed_tonnes_lag_8",
    "consumed_tonnes_rollmean_4", "consumed_tonnes_rollmean_8",
    "planned_pour_tonnes", "rain_mm", "avg_temp_c", "silo_capacity",
]
CATEGORICAL_FEATURES = ["behavior", "cement_type", "region", "site_id"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLS_FMT = "consumed_tonnes_t_plus_{h}"


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily data to weekly per site."""
    logger.info("Aggregating to weekly per site (%d rows)", len(df))
    weekly = (
        df.groupby("site_id")
        .resample("W", on="date")
        .agg({
            "consumed_tonnes": "sum",
            "planned_pour_tonnes": "sum",
            "rain_mm": "mean",
            "avg_temp_c": "mean",
            "behavior": "last",
            "cement_type": "last",
            "region": "last",
            "silo_capacity": "last",
        })
        .reset_index()
    )
    return weekly


def add_lag_features(
    df: pd.DataFrame,
    group_col: str = "site_id",
    target_col: str = "consumed_tonnes",
    lags: List[int] = None,
    rolling_windows: List[int] = None,
) -> pd.DataFrame:
    """Add lag and rolling-window features per group."""
    lags = lags or LAG_COLS
    rolling_windows = rolling_windows or ROLLING_WINDOWS
    df = df.copy().sort_values([group_col, "date"])

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df.groupby(group_col)[target_col].shift(lag)

    for window in rolling_windows:
        df[f"{target_col}_rollmean_{window}"] = (
            df.groupby(group_col)[target_col]
              .shift(1)
              .rolling(window=window, min_periods=1)
              .mean()
        )
    return df


def add_future_targets(
    df: pd.DataFrame,
    group_col: str = "site_id",
    target_col: str = "consumed_tonnes",
    horizon: int = None,
) -> pd.DataFrame:
    """Create target_t+1 ... target_t+horizon columns."""
    horizon = horizon or settings.HORIZON
    df = df.copy().sort_values([group_col, "date"])
    for h in range(1, horizon + 1):
        df[TARGET_COLS_FMT.format(h=h)] = (
            df.groupby(group_col)[target_col].shift(-h)
        )
    return df


def build_features(df: pd.DataFrame, horizon: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full preprocessing pipeline:
        daily -> weekly -> lag/rolling features -> multi-step targets.

    Returns:
        (weekly_with_features, model_df) where model_df has NaN rows dropped.
    """
    horizon = horizon or settings.HORIZON
    weekly = aggregate_weekly(df)
    weekly = add_lag_features(weekly)
    weekly = add_future_targets(weekly, horizon=horizon)
    model_df = weekly.dropna().copy()
    logger.info("Preprocessing complete: %d weekly rows, %d model rows", len(weekly), len(model_df))
    return weekly, model_df


def get_target_cols(horizon: int = None) -> List[str]:
    horizon = horizon or settings.HORIZON
    return [TARGET_COLS_FMT.format(h=h) for h in range(1, horizon + 1)]

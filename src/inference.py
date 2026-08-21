"""
Inference module: load the trained model and produce forecasts.

Supports:
  - Forecasting from a date in the dataset (uses site data for that date)
  - Recursive forecasting beyond the dataset (uses lag updates per step)
  - Optional exogenous scenario per week
"""
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

from src.common.config import settings
from src.common.logging_config import get_logger
from src.preprocessing import FEATURE_COLS

logger = get_logger(__name__)


class ForecastService:
    """Wraps the trained model and provides forecast methods."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path else settings.MODEL_PATH
        self.model = None
        self._weekly_df: Optional[pd.DataFrame] = None

    def load(self) -> "ForecastService":
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Train it first with `python -m src.pipeline`."
            )
        logger.info("Loading model from %s", self.model_path)
        self.model = joblib.load(self.model_path)
        return self

    def set_weekly_data(self, weekly_df: pd.DataFrame) -> "ForecastService":
        """Set the historical weekly data used for forecasts."""
        self._weekly_df = weekly_df.copy()
        self._weekly_df["date"] = pd.to_datetime(self._weekly_df["date"])
        return self

    @property
    def weekly_df(self) -> pd.DataFrame:
        if self._weekly_df is None:
            raise RuntimeError("No weekly data loaded. Call set_weekly_data() first.")
        return self._weekly_df

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def forecast_from_date(
        self,
        site_id: str,
        last_date: pd.Timestamp,
        horizon: int = None,
        feature_overrides: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Forecast horizon weeks starting after last_date (within dataset)."""
        horizon = horizon or settings.HORIZON
        site_data = self.weekly_df[self.weekly_df["site_id"] == site_id].sort_values("date")
        current_row = site_data[site_data["date"] == last_date]
        if current_row.empty:
            raise ValueError(f"No data for {site_id} at {last_date}")

        if feature_overrides:
            for k, v in feature_overrides.items():
                if k in current_row.columns:
                    current_row[k] = v

        y_future = self.model.predict(current_row[FEATURE_COLS])[0]
        future_dates = pd.date_range(
            start=last_date + timedelta(weeks=1),
            periods=horizon,
            freq="W",
        )
        return pd.DataFrame({
            "site_id": site_id,
            "date": future_dates,
            "forecast_consumed_tonnes": y_future[:horizon],
        })

    def forecast_future(
        self,
        site_id: str,
        start_date: pd.Timestamp,
        horizon: int = None,
        scenario: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Recursive forecast beyond the dataset."""
        horizon = horizon or settings.HORIZON
        site_data = self.weekly_df[self.weekly_df["site_id"] == site_id].sort_values("date")
        last_known_row = site_data.iloc[[-1]].copy()
        last_known_date = last_known_row["date"].iloc[0]

        if start_date <= last_known_date:
            return self.forecast_from_date(site_id, start_date, horizon)

        current_features = last_known_row[FEATURE_COLS].copy()
        current_date = last_known_date
        forecasts = []

        exog_lookup = {}
        if scenario is not None and not scenario.empty:
            exog_lookup = {
                pd.Timestamp(row["date"]): {
                    "planned_pour_tonnes": row["planned_pour_tonnes"],
                    "rain_mm": row["rain_mm"],
                    "avg_temp_c": row["avg_temp_c"],
                }
                for _, row in scenario.iterrows()
            }

        for h in range(1, horizon + 1):
            target_date = current_date + timedelta(weeks=1)

            if target_date in exog_lookup:
                ex = exog_lookup[target_date]
                current_features["planned_pour_tonnes"] = ex["planned_pour_tonnes"]
                current_features["rain_mm"] = ex["rain_mm"]
                current_features["avg_temp_c"] = ex["avg_temp_c"]

            y_pred = self.model.predict(current_features[FEATURE_COLS])[0]
            next_pred = y_pred[0]

            forecasts.append({
                "site_id": site_id,
                "date": target_date,
                "forecast_consumed_tonnes": next_pred,
            })

            # Update lags
            new_row = current_features.copy()
            new_row["consumed_tonnes_lag_8"] = new_row["consumed_tonnes_lag_4"]
            new_row["consumed_tonnes_lag_4"] = new_row["consumed_tonnes_lag_2"]
            new_row["consumed_tonnes_lag_2"] = new_row["consumed_tonnes_lag_1"]
            new_row["consumed_tonnes_lag_1"] = next_pred

            # Update rolling means from available lags
            lag_vals = [
                new_row["consumed_tonnes_lag_1"].iloc[0],
                new_row["consumed_tonnes_lag_2"].iloc[0],
                new_row["consumed_tonnes_lag_4"].iloc[0],
                new_row["consumed_tonnes_lag_8"].iloc[0],
            ]
            lag_vals = [v for v in lag_vals if not pd.isna(v)]
            if len(lag_vals) >= 4:
                new_row["consumed_tonnes_rollmean_4"] = np.mean(lag_vals[:4])
                new_row["consumed_tonnes_rollmean_8"] = np.mean(lag_vals)
            elif len(lag_vals) > 0:
                new_row["consumed_tonnes_rollmean_4"] = np.mean(lag_vals)
                new_row["consumed_tonnes_rollmean_8"] = np.mean(lag_vals)

            current_features = new_row
            current_date = target_date

        return pd.DataFrame(forecasts)

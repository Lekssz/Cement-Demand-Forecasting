"""
Production ARIMAX forecasting service.

This module is deliberately separate from the existing Random Forest
ForecastService so the team's RF work remains available as a comparison model.

Selected production forecasting model:
    ARIMAX(0, 1, 1)

External feature:
    planned_pour_tonnes

Forecast horizon:
    up to 8 weeks

The service fits one site model when first requested and caches the fitted
model for later requests during the same application session.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
ARIMA_ORDER = (0, 1, 1)
MAX_FORECAST_HORIZON = 8


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "operations_cleaned.csv"
)


# ---------------------------------------------------------
# 1. ARIMAX FORECAST SERVICE
# ---------------------------------------------------------
class ARIMAXForecastService:
    """
    Load cleaned data, build weekly site-level history,
    fit ARIMAX models, and generate future demand forecasts.
    """

    def __init__(
        self,
        data_path: Optional[Path] = None,
    ) -> None:

        self.data_path = (
            Path(data_path)
            if data_path
            else DEFAULT_DATA_PATH
        )

        self._daily_df: Optional[
            pd.DataFrame
        ] = None

        self._weekly_df: Optional[
            pd.DataFrame
        ] = None

        # Cache fitted statsmodels results by site.
        self._fitted_models: Dict[
            str,
            object,
        ] = {}


    # -----------------------------------------------------
    # 2. LOAD CLEANED DATA
    # -----------------------------------------------------
    def load_data(
        self,
    ) -> "ARIMAXForecastService":

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Cleaned data not found at "
                f"{self.data_path}"
            )

        daily_df = pd.read_csv(
            self.data_path
        )

        required_columns = {
            "date",
            "site_id",
            "consumed_tonnes",
            "planned_pour_tonnes",
        }

        missing = (
            required_columns
            - set(daily_df.columns)
        )

        if missing:
            raise ValueError(
                "Cleaned dataset is missing "
                f"required columns: "
                f"{sorted(missing)}"
            )

        daily_df["date"] = (
            pd.to_datetime(
                daily_df["date"]
            )
        )

        self._daily_df = (
            daily_df
            .sort_values(
                ["site_id", "date"]
            )
            .reset_index(drop=True)
        )

        self._weekly_df = (
            self._create_weekly_data(
                self._daily_df
            )
        )

        return self


    # -----------------------------------------------------
    # 3. CREATE WEEKLY DATA
    # -----------------------------------------------------
    @staticmethod
    def _create_weekly_data(
        daily_df: pd.DataFrame,
    ) -> pd.DataFrame:

        df = daily_df.copy()

        # Monday represents the beginning
        # of each modelling week.
        df["week_start"] = (
            df["date"]
            - pd.to_timedelta(
                df["date"].dt.weekday,
                unit="D",
            )
        )

        weekly_df = (
            df
            .groupby(
                [
                    "site_id",
                    "week_start",
                ],
                as_index=False,
            )
            .agg(
                consumed_tonnes=(
                    "consumed_tonnes",
                    "sum",
                ),
                planned_pour_tonnes=(
                    "planned_pour_tonnes",
                    "sum",
                ),
                days_in_week=(
                    "date",
                    "nunique",
                ),
            )
        )

        # Use only complete 7-day weeks,
        # matching the validated model.
        weekly_df = (
            weekly_df[
                weekly_df[
                    "days_in_week"
                ] == 7
            ]
            .copy()
        )

        return (
            weekly_df
            .sort_values(
                [
                    "site_id",
                    "week_start",
                ]
            )
            .reset_index(drop=True)
        )


    # -----------------------------------------------------
    # 4. DATA ACCESS
    # -----------------------------------------------------
    @property
    def weekly_df(
        self,
    ) -> pd.DataFrame:

        if self._weekly_df is None:
            raise RuntimeError(
                "No data loaded. "
                "Call load_data() first."
            )

        return self._weekly_df


    @property
    def daily_df(
        self,
    ) -> pd.DataFrame:

        if self._daily_df is None:
            raise RuntimeError(
                "No data loaded. "
                "Call load_data() first."
            )

        return self._daily_df


    def list_sites(
        self,
    ) -> list:

        return sorted(
            self.weekly_df[
                "site_id"
            ]
            .unique()
            .tolist()
        )


    # -----------------------------------------------------
    # 5. FIT ONE SITE MODEL
    # -----------------------------------------------------
    def fit_site(
        self,
        site_id: str,
        force_refit: bool = False,
    ):
        """
        Fit ARIMAX(0,1,1) for one site.

        If the site has already been fitted during this
        application session, return the cached model.
        """

        if (
            not force_refit
            and site_id
            in self._fitted_models
        ):
            return (
                self._fitted_models[
                    site_id
                ]
            )

        site_data = (
            self.weekly_df[
                self.weekly_df[
                    "site_id"
                ] == site_id
            ]
            .sort_values(
                "week_start"
            )
            .reset_index(drop=True)
        )

        if site_data.empty:
            raise ValueError(
                f"Unknown site_id: {site_id}"
            )

        target = (
            site_data[
                "consumed_tonnes"
            ]
            .astype(float)
            .reset_index(drop=True)
        )

        exog = (
            site_data[
                [
                    "planned_pour_tonnes"
                ]
            ]
            .astype(float)
            .reset_index(drop=True)
        )

        model = SARIMAX(
            endog=target,
            exog=exog,
            order=ARIMA_ORDER,
            seasonal_order=(
                0, 0, 0, 0
            ),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        fitted_model = (
            model.fit(
                disp=False
            )
        )

        self._fitted_models[
            site_id
        ] = fitted_model

        return fitted_model


    # -----------------------------------------------------
    # 6. FORECAST FUTURE WEEKS
    # -----------------------------------------------------
    def forecast_site(
        self,
        site_id: str,
        planned_pour_tonnes: Sequence[
            float
        ],
    ) -> pd.DataFrame:
        """
        Forecast future weekly cement demand for one site.

        planned_pour_tonnes must contain between 1 and 8
        weekly planned-pour values.

        The planned schedule is treated as known at
        forecast time, matching the validated project
        assumption.
        """

        planned_pours = np.asarray(
            planned_pour_tonnes,
            dtype=float,
        )

        horizon = len(
            planned_pours
        )

        if horizon < 1:
            raise ValueError(
                "At least one future planned "
                "pour value is required."
            )

        if (
            horizon
            > MAX_FORECAST_HORIZON
        ):
            raise ValueError(
                "Forecast horizon cannot exceed "
                f"{MAX_FORECAST_HORIZON} weeks."
            )

        if (
            planned_pours < 0
        ).any():
            raise ValueError(
                "planned_pour_tonnes cannot "
                "contain negative values."
            )

        fitted_model = (
            self.fit_site(
                site_id
            )
        )

        future_exog = pd.DataFrame({
            "planned_pour_tonnes": (
                planned_pours
            )
        })

        forecast = (
            fitted_model.forecast(
                steps=horizon,
                exog=future_exog,
            )
        )

        forecast_values = np.maximum(
            np.asarray(
                forecast,
                dtype=float,
            ),
            0.0,
        )

        site_history = (
            self.weekly_df[
                self.weekly_df[
                    "site_id"
                ] == site_id
            ]
            .sort_values(
                "week_start"
            )
        )

        last_week_start = (
            site_history[
                "week_start"
            ].max()
        )

        future_week_starts = (
            pd.date_range(
                start=(
                    last_week_start
                    + pd.Timedelta(
                        weeks=1
                    )
                ),
                periods=horizon,
                freq="W-MON",
            )
        )

        return pd.DataFrame({
            "site_id": site_id,
            "week_start": (
                future_week_starts
            ),
            "planned_pour_tonnes": (
                planned_pours
            ),
            "forecast_consumed_tonnes": (
                forecast_values
            ),
        })


    # -----------------------------------------------------
    # 7. CLEAR MODEL CACHE
    # -----------------------------------------------------
    def clear_cache(
        self,
        site_id: Optional[str] = None,
    ) -> None:
        """
        Clear one cached site model or all cached models.

        Useful after retraining data is updated.
        """

        if site_id is None:
            self._fitted_models.clear()
            return

        self._fitted_models.pop(
            site_id,
            None,
        )

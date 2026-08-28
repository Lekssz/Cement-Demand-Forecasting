"""
End-to-end forecasting + inventory recommendation service.

Flow:
    future daily planned-pour schedule
        -> aggregate to weekly planned pours
        -> ARIMAX weekly demand forecast
        -> distribute weekly forecast back to days
        -> 3-day inventory recommendation

This keeps forecasting and inventory logic separate while providing one
production-friendly function for the API/dashboard.
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.arimax_forecaster import ARIMAXForecastService
from src.inventory_optimizer import (
    DEFAULT_COVERAGE_DAYS,
    calculate_inventory_recommendation,
)


# ---------------------------------------------------------
# 1. END-TO-END SERVICE
# ---------------------------------------------------------
class ForecastInventoryService:
    """
    Combine the production ARIMAX forecaster with the
    validated inventory recommendation policy.
    """

    def __init__(
        self,
        forecast_service: Optional[
            ARIMAXForecastService
        ] = None,
    ) -> None:

        self.forecast_service = (
            forecast_service
            if forecast_service is not None
            else ARIMAXForecastService()
        )


    # -----------------------------------------------------
    # 2. LOAD FORECASTING DATA
    # -----------------------------------------------------
    def load(
        self,
    ) -> "ForecastInventoryService":

        self.forecast_service.load_data()

        return self


    # -----------------------------------------------------
    # 3. VALIDATE DAILY POUR SCHEDULE
    # -----------------------------------------------------
    @staticmethod
    def _prepare_daily_schedule(
        daily_plan: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate the future daily planned-pour schedule.

        Required columns:
            date
            planned_pour_tonnes

        Missing calendar days are not silently invented.
        The supplied schedule must contain consecutive days.
        """

        required_columns = {
            "date",
            "planned_pour_tonnes",
        }

        missing = (
            required_columns
            - set(daily_plan.columns)
        )

        if missing:
            raise ValueError(
                "daily_plan is missing required "
                f"columns: {sorted(missing)}"
            )

        schedule = daily_plan[
            [
                "date",
                "planned_pour_tonnes",
            ]
        ].copy()

        schedule["date"] = pd.to_datetime(
            schedule["date"]
        )

        schedule[
            "planned_pour_tonnes"
        ] = pd.to_numeric(
            schedule[
                "planned_pour_tonnes"
            ],
            errors="raise",
        )

        schedule = (
            schedule
            .sort_values("date")
            .reset_index(drop=True)
        )

        if schedule.empty:
            raise ValueError(
                "daily_plan cannot be empty."
            )

        if (
            schedule[
                "planned_pour_tonnes"
            ] < 0
        ).any():
            raise ValueError(
                "planned_pour_tonnes cannot "
                "contain negative values."
            )

        if schedule["date"].duplicated().any():
            raise ValueError(
                "daily_plan contains duplicate dates."
            )

        # The inventory policy needs at least three future
        # daily forecasts for the validated 3-day target.
        if len(schedule) < DEFAULT_COVERAGE_DAYS:
            raise ValueError(
                "daily_plan must contain at least "
                f"{DEFAULT_COVERAGE_DAYS} days."
            )

        # ARIMAX was validated for a maximum 8-week horizon.
        if len(schedule) > 56:
            raise ValueError(
                "daily_plan cannot exceed 56 days "
                "(8 weeks)."
            )

        # Require consecutive dates so that 'next 3 days'
        # really means three consecutive calendar days.
        expected_dates = pd.date_range(
            start=schedule["date"].iloc[0],
            periods=len(schedule),
            freq="D",
        )

        if not schedule["date"].reset_index(
            drop=True
        ).equals(
            pd.Series(expected_dates)
        ):
            raise ValueError(
                "daily_plan dates must be "
                "consecutive calendar days."
            )

        return schedule


    # -----------------------------------------------------
    # 4. CONVERT DAILY PLAN TO WEEKLY ARIMAX INPUT
    # -----------------------------------------------------
    @staticmethod
    def _create_weekly_plan(
        daily_plan: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aggregate the daily planned-pour schedule into
        weekly totals for ARIMAX.
        """

        schedule = daily_plan.copy()

        schedule["week_start"] = (
            schedule["date"]
            - pd.to_timedelta(
                schedule["date"].dt.weekday,
                unit="D",
            )
        )

        weekly_plan = (
            schedule
            .groupby(
                "week_start",
                as_index=False,
            )
            .agg(
                planned_pour_tonnes=(
                    "planned_pour_tonnes",
                    "sum",
                ),
                days_supplied=(
                    "date",
                    "nunique",
                ),
            )
            .sort_values("week_start")
            .reset_index(drop=True)
        )

        # To match the validated weekly model, each forecast
        # week must contain all seven calendar days.
        incomplete = weekly_plan[
            weekly_plan[
                "days_supplied"
            ] != 7
        ]

        if not incomplete.empty:
            raise ValueError(
                "Each forecast week must contain "
                "7 daily schedule rows. Start the "
                "schedule on a Monday and supply "
                "complete weeks."
            )

        if len(weekly_plan) > 8:
            raise ValueError(
                "Weekly plan exceeds the "
                "8-week forecast horizon."
            )

        return weekly_plan


    # -----------------------------------------------------
    # 5. DISTRIBUTE WEEKLY FORECAST BACK TO DAYS
    # -----------------------------------------------------
    @staticmethod
    def _create_daily_forecast(
        daily_plan: pd.DataFrame,
        weekly_forecast: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Allocate each weekly ARIMAX forecast to days using
        the planned-pour proportions within that week.

        If the planned total for a week is zero, the weekly
        forecast is split equally across seven days.
        """

        daily = daily_plan.copy()

        daily["week_start"] = (
            daily["date"]
            - pd.to_timedelta(
                daily["date"].dt.weekday,
                unit="D",
            )
        )

        daily = daily.merge(
            weekly_forecast[
                [
                    "week_start",
                    "forecast_consumed_tonnes",
                ]
            ],
            on="week_start",
            how="left",
            validate="many_to_one",
        )

        if (
            daily[
                "forecast_consumed_tonnes"
            ].isna().any()
        ):
            raise ValueError(
                "Some daily schedule rows could "
                "not be matched to an ARIMAX "
                "weekly forecast."
            )

        weekly_planned = (
            daily
            .groupby("week_start")[
                "planned_pour_tonnes"
            ]
            .transform("sum")
        )

        daily["forecast_weight"] = np.where(
            weekly_planned > 0,
            (
                daily[
                    "planned_pour_tonnes"
                ]
                / weekly_planned
            ),
            1.0 / 7.0,
        )

        daily[
            "forecast_consumed_tonnes"
        ] = (
            daily[
                "forecast_consumed_tonnes"
            ]
            * daily[
                "forecast_weight"
            ]
        )

        return daily[
            [
                "date",
                "week_start",
                "planned_pour_tonnes",
                "forecast_consumed_tonnes",
            ]
        ].copy()


    # -----------------------------------------------------
    # 6. FORECAST + INVENTORY RECOMMENDATION
    # -----------------------------------------------------
    def recommend(
        self,
        site_id: str,
        daily_plan: pd.DataFrame,
        current_inventory_tonnes: float,
        silo_capacity: float,
        safety_stock_tonnes: float,
        coverage_days: int = DEFAULT_COVERAGE_DAYS,
    ) -> dict:
        """
        Generate the future demand forecast and current
        inventory recommendation for one site.
        """

        schedule = (
            self._prepare_daily_schedule(
                daily_plan
            )
        )

        weekly_plan = (
            self._create_weekly_plan(
                schedule
            )
        )

        weekly_forecast = (
            self.forecast_service.forecast_site(
                site_id=site_id,
                planned_pour_tonnes=(
                    weekly_plan[
                        "planned_pour_tonnes"
                    ].tolist()
                ),
            )
        )

        # Guard against a schedule that does not begin
        # on the same week the ARIMAX service is forecasting.
        expected_first_week = (
            weekly_forecast[
                "week_start"
            ].iloc[0]
        )

        supplied_first_week = (
            weekly_plan[
                "week_start"
            ].iloc[0]
        )

        if (
            supplied_first_week
            != expected_first_week
        ):
            raise ValueError(
                "daily_plan starts on "
                f"{supplied_first_week.date()}, "
                "but the next ARIMAX forecast week "
                "starts on "
                f"{expected_first_week.date()}."
            )

        daily_forecast = (
            self._create_daily_forecast(
                daily_plan=schedule,
                weekly_forecast=(
                    weekly_forecast
                ),
            )
        )

        next_daily_demand = (
            daily_forecast[
                "forecast_consumed_tonnes"
            ]
            .head(coverage_days)
            .tolist()
        )

        inventory = (
            calculate_inventory_recommendation(
                current_inventory_tonnes=(
                    current_inventory_tonnes
                ),
                silo_capacity=(
                    silo_capacity
                ),
                daily_forecast_tonnes=(
                    next_daily_demand
                ),
                safety_stock_tonnes=(
                    safety_stock_tonnes
                ),
                coverage_days=(
                    coverage_days
                ),
            )
        )

        return {
            "site_id": site_id,
            "weekly_forecast": (
                weekly_forecast
            ),
            "daily_forecast": (
                daily_forecast
            ),
            "inventory_recommendation": (
                inventory
            ),
        }

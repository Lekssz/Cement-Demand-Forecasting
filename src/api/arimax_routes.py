"""
FastAPI routes for the selected production ARIMAX model and
forecast-driven inventory recommendation.

The existing Random Forest routes remain untouched.
"""

from datetime import datetime
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.forecast_inventory_service import ForecastInventoryService


router = APIRouter(
    prefix="/arimax",
    tags=["ARIMAX Production"],
)


# ---------------------------------------------------------
# VALIDATED SITE SAFETY STOCK
# ---------------------------------------------------------
# 98th percentile weekly ARIMAX under-forecast error used by
# the validated inventory policy.
SAFETY_STOCK_BY_SITE = {
    "SITE_001": 71.51,
    "SITE_002": 6.43,
    "SITE_003": 79.28,
    "SITE_004": 1.46,
    "SITE_005": 50.67,
    "SITE_006": 40.87,
    "SITE_007": 98.11,
    "SITE_008": 64.74,
    "SITE_009": 6.74,
    "SITE_010": 67.82,
    "SITE_011": 62.98,
    "SITE_012": 3.20,
    "SITE_013": 27.03,
    "SITE_014": 26.50,
    "SITE_015": 2.08,
    "SITE_016": 43.12,
    "SITE_017": 62.70,
    "SITE_018": 53.92,
    "SITE_019": 2.49,
    "SITE_020": 54.88,
    "SITE_021": 60.74,
    "SITE_022": 44.84,
    "SITE_023": 2.99,
    "SITE_024": 31.86,
    "SITE_025": 83.81,
    "SITE_026": 45.79,
    "SITE_027": 4.06,
    "SITE_028": 51.44,
    "SITE_029": 2.04,
    "SITE_030": 42.80,
}


# ---------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS
# ---------------------------------------------------------
class DailyPlanPoint(BaseModel):
    date: datetime
    planned_pour_tonnes: float = Field(ge=0)


class ForecastInventoryRequest(BaseModel):
    site_id: str
    current_inventory_tonnes: float = Field(ge=0)

    # Optional for backward compatibility.
    # If omitted, the API resolves them from the project data/policy.
    silo_capacity: Optional[float] = Field(default=None, gt=0)
    safety_stock_tonnes: Optional[float] = Field(default=None, ge=0)

    daily_plan: List[DailyPlanPoint]


class WeeklyForecastPoint(BaseModel):
    site_id: str
    week_start: datetime
    planned_pour_tonnes: float
    forecast_consumed_tonnes: float


class DailyForecastPoint(BaseModel):
    date: datetime
    week_start: datetime
    planned_pour_tonnes: float
    forecast_consumed_tonnes: float


class InventoryRecommendation(BaseModel):
    current_inventory_tonnes: float
    silo_capacity: float
    inventory_utilisation_pct: float
    today_expected_demand_tonnes: float
    coverage_days: int
    coverage_forecast_tonnes: float
    safety_stock_tonnes: float
    reorder_trigger_tonnes: float
    target_inventory_tonnes: float
    reorder_alert: bool
    recommended_order_tonnes: float
    projected_inventory_after_order_tonnes: float
    remaining_capacity_after_order_tonnes: float


class ForecastInventoryResponse(BaseModel):
    site_id: str
    model: str
    weekly_forecast: List[WeeklyForecastPoint]
    daily_forecast: List[DailyForecastPoint]
    inventory_recommendation: InventoryRecommendation
    generated_at: datetime


# ---------------------------------------------------------
# SERVICE
# ---------------------------------------------------------
_service: Optional[ForecastInventoryService] = None


def get_service() -> ForecastInventoryService:
    global _service

    if _service is None:
        _service = ForecastInventoryService().load()

    return _service


# ---------------------------------------------------------
# SITE CONFIGURATION
# ---------------------------------------------------------
def resolve_site_config(site_id: str) -> dict:
    service = get_service()

    if site_id not in service.forecast_service.list_sites():
        raise ValueError(f"Unknown site_id: {site_id}")

    daily = service.forecast_service.daily_df
    weekly = service.forecast_service.weekly_df

    site_daily = (
        daily[daily["site_id"] == site_id]
        .sort_values("date")
    )

    site_weekly = (
        weekly[weekly["site_id"] == site_id]
        .sort_values("week_start")
    )

    capacity = float(
        site_daily["silo_capacity"].iloc[-1]
    )

    next_forecast_week = (
        pd.Timestamp(site_weekly["week_start"].max())
        + pd.Timedelta(weeks=1)
    )

    safety_stock = SAFETY_STOCK_BY_SITE.get(site_id)

    if safety_stock is None:
        raise ValueError(
            f"No validated safety stock configured for {site_id}"
        )

    return {
        "site_id": site_id,
        "silo_capacity": capacity,
        "safety_stock_tonnes": float(safety_stock),
        "next_forecast_week": next_forecast_week,
    }


# ---------------------------------------------------------
# API ROUTES
# ---------------------------------------------------------
@router.get("/health")
def arimax_health():
    try:
        service = get_service()

        return {
            "status": "ok",
            "model": "ARIMAX(0,1,1)",
            "forecast_horizon_weeks": 8,
            "sites_loaded": len(
                service.forecast_service.list_sites()
            ),
            "inventory_policy": "3-day order-up-to",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )


@router.get("/sites")
def arimax_sites():
    try:
        return get_service().forecast_service.list_sites()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )


@router.get("/sites/{site_id}/config")
def arimax_site_config(site_id: str):
    try:
        config = resolve_site_config(site_id)

        return {
            **config,
            "next_forecast_week": (
                config["next_forecast_week"].isoformat()
            ),
            "model": "ARIMAX(0,1,1)",
            "inventory_policy": "3-day order-up-to",
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )


@router.post(
    "/forecast-inventory",
    response_model=ForecastInventoryResponse,
)
def forecast_inventory(
    req: ForecastInventoryRequest,
) -> ForecastInventoryResponse:

    try:
        service = get_service()
        config = resolve_site_config(req.site_id)

        capacity = (
            float(req.silo_capacity)
            if req.silo_capacity is not None
            else config["silo_capacity"]
        )

        safety_stock = (
            float(req.safety_stock_tonnes)
            if req.safety_stock_tonnes is not None
            else config["safety_stock_tonnes"]
        )

        daily_plan = pd.DataFrame(
            [
                {
                    "date": point.date,
                    "planned_pour_tonnes": point.planned_pour_tonnes,
                }
                for point in req.daily_plan
            ]
        )

        result = service.recommend(
            site_id=req.site_id,
            daily_plan=daily_plan,
            current_inventory_tonnes=req.current_inventory_tonnes,
            silo_capacity=capacity,
            safety_stock_tonnes=safety_stock,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    weekly_points = [
        WeeklyForecastPoint(
            site_id=row.site_id,
            week_start=row.week_start.to_pydatetime(),
            planned_pour_tonnes=float(row.planned_pour_tonnes),
            forecast_consumed_tonnes=float(row.forecast_consumed_tonnes),
        )
        for row in result["weekly_forecast"].itertuples()
    ]

    daily_points = [
        DailyForecastPoint(
            date=row.date.to_pydatetime(),
            week_start=row.week_start.to_pydatetime(),
            planned_pour_tonnes=float(row.planned_pour_tonnes),
            forecast_consumed_tonnes=float(row.forecast_consumed_tonnes),
        )
        for row in result["daily_forecast"].itertuples()
    ]

    return ForecastInventoryResponse(
        site_id=req.site_id,
        model="ARIMAX(0,1,1)",
        weekly_forecast=weekly_points,
        daily_forecast=daily_points,
        inventory_recommendation=InventoryRecommendation(
            **result["inventory_recommendation"]
        ),
        generated_at=datetime.utcnow(),
    )

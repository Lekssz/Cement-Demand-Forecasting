"""
FastAPI service for cement demand forecasting inference.
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.common.config import settings
from src.common.logging_config import get_logger
from src.common.schemas import (
    FeatureRow,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    SiteInfo,
)
from src.data_loader import load_or_generate_demo
from src.inference import ForecastService
from src.preprocessing import build_features, FEATURE_COLS

logger = get_logger(__name__)

app = FastAPI(
    title="Cement Demand Forecasting API",
    description="Predict 8-week cement demand per site using a Random Forest model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service: Optional[ForecastService] = None


@app.on_event("startup")
def startup() -> None:
    global service
    logger.info("Starting API service...")
    service = ForecastService()

    if service.model_path.exists():
        service.load()
    else:
        logger.warning("Model not found at %s. Train first.", service.model_path)

    # Load historical data for forecasting context
    try:
        df = load_or_generate_demo()
        _, weekly = build_features(df)
        service.set_weekly_data(weekly)
        logger.info("Loaded weekly data for %d sites", weekly["site_id"].nunique())
    except Exception as e:
        logger.warning("Could not load historical data: %s", e)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=service is not None and service.is_loaded,
        model_path=str(service.model_path) if service else "",
    )


@app.get("/sites", response_model=List[str])
def list_sites() -> List[str]:
    if service is None or service.weekly_df is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return sorted(service.weekly_df["site_id"].unique().tolist())


@app.get("/sites/{site_id}/info", response_model=SiteInfo)
def get_site_info(site_id: str) -> SiteInfo:
    """Get site metadata and last known feature row."""
    if service is None or service.weekly_df is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    site_data = service.weekly_df[service.weekly_df["site_id"] == site_id]
    if site_data.empty:
        raise HTTPException(status_code=404, detail=f"Site {site_id} not found")

    site_data = site_data.sort_values("date")
    last_row = site_data.iloc[-1]
    last_date = last_row["date"]

    return SiteInfo(
        site_id=site_id,
        region=last_row["region"],
        cement_type=last_row["cement_type"],
        behavior=last_row["behavior"],
        silo_capacity=float(last_row["silo_capacity"]),
        last_date_in_data=last_date.to_pydatetime(),
        last_feature_row=FeatureRow(
            site_id=site_id,
            date=last_date.to_pydatetime(),
            consumed_tonnes_lag_1=float(last_row["consumed_tonnes_lag_1"]),
            consumed_tonnes_lag_2=float(last_row["consumed_tonnes_lag_2"]),
            consumed_tonnes_lag_4=float(last_row["consumed_tonnes_lag_4"]),
            consumed_tonnes_lag_8=float(last_row["consumed_tonnes_lag_8"]),
            consumed_tonnes_rollmean_4=float(last_row["consumed_tonnes_rollmean_4"]),
            consumed_tonnes_rollmean_8=float(last_row["consumed_tonnes_rollmean_8"]),
            planned_pour_tonnes=float(last_row["planned_pour_tonnes"]),
            rain_mm=float(last_row["rain_mm"]),
            avg_temp_c=float(last_row["avg_temp_c"]),
            silo_capacity=float(last_row["silo_capacity"]),
            behavior=last_row["behavior"],
            cement_type=last_row["cement_type"],
            region=last_row["region"],
        ),
    )


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    if service is None or not service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_date = pd.Timestamp(req.start_date)
    horizon = min(req.horizon, settings.HORIZON)

    # Build scenario DataFrame if provided
    scenario_df = None
    if req.scenario:
        scenario_df = pd.DataFrame(req.scenario)
        if "date" in scenario_df.columns:
            scenario_df["date"] = pd.to_datetime(scenario_df["date"])

    try:
        site_data = service.weekly_df[service.weekly_df["site_id"] == req.site_id]
        if site_data.empty:
            raise HTTPException(status_code=404, detail=f"Site {req.site_id} not found")

        site_data = site_data.sort_values("date")
        last_known_date = site_data["date"].max()

        # Determine forecast mode
        if req.mode == "from_last_date":
            # Use last date in data, with feature overrides
            forecast_df = service.forecast_from_date(
                req.site_id, last_known_date, horizon,
                feature_overrides=req.feature_overrides,
            )
        elif req.mode == "custom_date":
            # Use specific date from dataset
            if start_date > last_known_date:
                raise HTTPException(
                    status_code=400,
                    detail=f"custom_date mode requires start_date <= last known date ({last_known_date.date()})"
                )
            forecast_df = service.forecast_from_date(
                req.site_id, start_date, horizon,
                feature_overrides=req.feature_overrides,
            )
        else:  # future_dates
            forecast_df = service.forecast_future(
                req.site_id, start_date, horizon, scenario=scenario_df,
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Forecast failed")
        raise HTTPException(status_code=500, detail=str(e))

    points = [
        ForecastPoint(
            site_id=row.site_id,
            date=row.date.to_pydatetime(),
            forecast_consumed_tonnes=float(row.forecast_consumed_tonnes),
        )
        for row in forecast_df.itertuples()
    ]

    return ForecastResponse(
        site_id=req.site_id,
        horizon=horizon,
        forecasts=points,
        generated_at=datetime.utcnow(),
    )

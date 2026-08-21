"""
Pydantic schemas shared between the API and training pipeline.
"""
from datetime import datetime
from typing import List, Optional, Literal

from pydantic import BaseModel, Field


# -- Feature row ----------------------------------------------------------
class FeatureRow(BaseModel):
    """A single feature row used for inference."""
    site_id: str
    date: datetime
    consumed_tonnes_lag_1: float
    consumed_tonnes_lag_2: float
    consumed_tonnes_lag_4: float
    consumed_tonnes_lag_8: float
    consumed_tonnes_rollmean_4: float
    consumed_tonnes_rollmean_8: float
    planned_pour_tonnes: float
    rain_mm: float
    avg_temp_c: float
    silo_capacity: float
    behavior: str
    cement_type: str
    region: str


# -- Forecast request -----------------------------------------------------
class ForecastRequest(BaseModel):
    site_id: str
    start_date: datetime
    horizon: int = Field(default=8, ge=1, le=52)
    # Forecast mode matching Streamlit app
    mode: Literal["from_last_date", "custom_date", "future_dates"] = "from_last_date"
    # Optional exogenous scenario; one row per forecast week
    scenario: Optional[List[dict]] = None
    # Override the feature row used at start_date (lags, rolling, etc.)
    feature_overrides: Optional[dict] = None


# -- Forecast response ----------------------------------------------------
class ForecastPoint(BaseModel):
    site_id: str
    date: datetime
    forecast_consumed_tonnes: float


class ForecastResponse(BaseModel):
    site_id: str
    horizon: int
    forecasts: List[ForecastPoint]
    generated_at: datetime


# -- Site info ------------------------------------------------------------
class SiteInfo(BaseModel):
    site_id: str
    region: str
    cement_type: str
    behavior: str
    silo_capacity: float
    last_date_in_data: datetime
    last_feature_row: FeatureRow


# -- Health ---------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str

"""
Centralized configuration loaded from environment variables.
"""
import os
from pathlib import Path
from typing import Optional


class Settings:
    # Paths
    PROJECT_ROOT: Path = Path(os.getenv("PROJECT_ROOT", "/app"))
    DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
    MODEL_DIR: Path = PROJECT_ROOT / "models"
    MODEL_PATH: Path = MODEL_DIR / "cement_demand_rf.pkl"

    # Service ports (overridable via env)
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    DASHBOARD_PORT: int = int(os.getenv("DASHBOARD_PORT", "8501"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # MLflow
    MLFLOW_TRACKING_URI: Optional[str] = os.getenv("MLFLOW_TRACKING_URI")
    MLFLOW_EXPERIMENT: str = os.getenv("MLFLOW_EXPERIMENT", "cement_demand_forecasting")

    # Forecast defaults
    HORIZON: int = int(os.getenv("HORIZON", "8"))

    # Make sure model dir exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()

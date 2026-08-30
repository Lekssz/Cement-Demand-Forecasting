"""
API routes for model evaluation results.

These routes expose stored backtest results used by the
dashboard to explain model performance and model selection.
"""

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException


# ---------------------------------------------------------
# 1. ROUTER
# ---------------------------------------------------------
# All evaluation endpoints will begin with:
#
# /model-evaluation/...
#
# This keeps model evaluation separate from the production
# forecasting endpoints such as /arimax/forecast-inventory.
router = APIRouter(
    prefix="/model-evaluation",
    tags=["Model Evaluation"],
)


# ---------------------------------------------------------
# 2. PROJECT PATHS
# ---------------------------------------------------------
# This file lives at:
#
# project_root/src/api/model_evaluation_routes.py
#
# Therefore parents[2] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"


# ---------------------------------------------------------
# 3. OVERALL BACKTEST REPORTS
# ---------------------------------------------------------
# These contain the five overall 8-week backtest windows
# for ARIMAX and Random Forest.
ARIMAX_BACKTEST_PATH = (
    REPORTS_DIR
    / "arimax_backtest_results.csv"
)

RF_BACKTEST_PATH = (
    REPORTS_DIR
    / "rf_backtest_results.csv"
)


# ---------------------------------------------------------
# 4. SITE PERFORMANCE REPORTS
# ---------------------------------------------------------
# These contain the overall forecasting performance for
# each of the 30 construction sites.
ARIMAX_SITE_SUMMARY_PATH = (
    REPORTS_DIR
    / "arimax_site_summary.csv"
)

RF_SITE_SUMMARY_PATH = (
    REPORTS_DIR
    / "rf_site_summary.csv"
)


# ---------------------------------------------------------
# 5. ARIMAX DETAILED BACKTEST REPORTS
# ---------------------------------------------------------
# These files allow us to inspect ARIMAX more deeply:
#
# - performance for every site in every backtest window
# - actual vs forecast demand for each site and window
ARIMAX_SITE_WINDOW_PATH = (
    REPORTS_DIR
    / "arimax_site_window_results.csv"
)

ARIMAX_PREDICTIONS_PATH = (
    REPORTS_DIR
    / "arimax_backtest_predictions.csv"
)


# ---------------------------------------------------------
# 6. OVERALL MODEL COMPARISON
# ---------------------------------------------------------
# Calculate one mean MAPE for each model across the same
# five 8-week backtest periods.
#
# This is the high-level comparison used to explain why
# ARIMAX was selected as the production model.
@router.get("/summary")
def model_comparison_summary():

    try:
        arimax_df = pd.read_csv(
            ARIMAX_BACKTEST_PATH
        )

        rf_df = pd.read_csv(
            RF_BACKTEST_PATH
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Evaluation report not found: {exc}"
            ),
        )

    arimax_mean_mape = float(
        arimax_df["MAPE"].mean()
    )

    rf_mean_mape = float(
        rf_df["MAPE"].mean()
    )

    return {
        "metric": "MAPE",
        "target_pct": 15.0,
        "models": [
            {
                "model": "ARIMAX",
                "mean_mape": round(
                    arimax_mean_mape,
                    2,
                ),
                "meets_target": bool(
                    arimax_mean_mape <= 15
                ),
                "role": "Production",
            },
            {
                "model": "Random Forest",
                "mean_mape": round(
                    rf_mean_mape,
                    2,
                ),
                "meets_target": bool(
                    rf_mean_mape <= 15
                ),
                "role": "Benchmark",
            },
        ],
        "selected_model": "ARIMAX",
    }


# ---------------------------------------------------------
# 7. SITE-LEVEL MODEL PERFORMANCE
# ---------------------------------------------------------
# Return site-level MAE, RMSE and MAPE for either:
#
# /model-evaluation/sites/arimax
#
# or
#
# /model-evaluation/sites/random-forest
#
# Although the source CSVs use slightly different column
# names, this endpoint normalises both into the same format
# for the dashboard.
@router.get("/sites/{model_name}")
def site_performance(
    model_name: str,
):

    model_key = (
        model_name
        .strip()
        .lower()
    )

    if model_key == "arimax":

        path = ARIMAX_SITE_SUMMARY_PATH
        display_name = "ARIMAX"
        mape_column = "overall_MAPE"

    elif model_key in {
        "random-forest",
        "random_forest",
        "rf",
    }:

        path = RF_SITE_SUMMARY_PATH
        display_name = "Random Forest"
        mape_column = "MAPE"

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "Model must be 'arimax' "
                "or 'random-forest'."
            ),
        )

    try:
        df = pd.read_csv(
            path
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Site report not found: {path}"
            ),
        )

    sites = []

    for _, row in df.iterrows():

        mape = float(
            row[mape_column]
        )

        sites.append(
            {
                "site_id": str(
                    row["site_id"]
                ),
                "mae": round(
                    float(row["MAE"]),
                    2,
                ),
                "rmse": round(
                    float(row["RMSE"]),
                    2,
                ),
                "mape": round(
                    mape,
                    2,
                ),
                "meets_target": bool(
                    mape <= 15
                ),
            }
        )

    # Sort the output by MAPE so the dashboard receives
    # sites from best to worst forecasting performance.
    sites = sorted(
        sites,
        key=lambda site: site["mape"],
    )

    sites_meeting_target = sum(
        site["meets_target"]
        for site in sites
    )

    return {
        "model": display_name,
        "target_pct": 15.0,
        "sites_meeting_target": int(
            sites_meeting_target
        ),
        "sites_above_target": (
            len(sites)
            - sites_meeting_target
        ),
        "total_sites": len(sites),
        "sites": sites,
    }


# ---------------------------------------------------------
# 8. ARIMAX OVERALL BACKTEST WINDOWS
# ---------------------------------------------------------
# Return the five expanding 8-week ARIMAX evaluation
# windows used during model validation.
#
# This allows the dashboard to show whether performance
# remained stable through different periods rather than
# relying only on one final test.
@router.get("/arimax/windows")
def arimax_backtest_windows():

    try:
        df = pd.read_csv(
            ARIMAX_BACKTEST_PATH
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=(
                "ARIMAX backtest report "
                "not found."
            ),
        )

    windows = []

    for _, row in df.iterrows():

        mape = float(
            row["MAPE"]
        )

        windows.append(
            {
                "window": int(
                    row["window"]
                ),
                "start_date": str(
                    row["start_date"]
                ),
                "end_date": str(
                    row["end_date"]
                ),
                "mae": round(
                    float(row["MAE"]),
                    2,
                ),
                "rmse": round(
                    float(row["RMSE"]),
                    2,
                ),
                "mape": round(
                    mape,
                    2,
                ),
                "meets_target": bool(
                    mape <= 15
                ),
            }
        )

    windows = sorted(
        windows,
        key=lambda item: item["window"],
    )

    return {
        "model": "ARIMAX",
        "target_pct": 15.0,
        "windows": windows,
    }


# ---------------------------------------------------------
# 9. ARIMAX BACKTEST WINDOWS FOR ONE SITE
# ---------------------------------------------------------
# Return how one particular site performed in each of the
# five ARIMAX backtest periods.
#
# Example:
#
# /model-evaluation/arimax/site/SITE_003/windows
@router.get(
    "/arimax/site/{site_id}/windows"
)
def arimax_site_windows(
    site_id: str,
):

    try:
        df = pd.read_csv(
            ARIMAX_SITE_WINDOW_PATH
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=(
                "ARIMAX site-window report "
                "not found."
            ),
        )

    site_id = (
        site_id
        .strip()
        .upper()
    )

    site_df = df[
        df["site_id"] == site_id
    ].copy()

    if site_df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Site {site_id} not found."
            ),
        )

    results = []

    for _, row in site_df.iterrows():

        mape = float(
            row["MAPE"]
        )

        results.append(
            {
                "window": int(
                    row["window"]
                ),
                "mae": round(
                    float(row["MAE"]),
                    2,
                ),
                "rmse": round(
                    float(row["RMSE"]),
                    2,
                ),
                "mape": round(
                    mape,
                    2,
                ),
                "meets_target": bool(
                    mape <= 15
                ),
            }
        )

    results = sorted(
        results,
        key=lambda item: item["window"],
    )

    return {
        "model": "ARIMAX",
        "site_id": site_id,
        "target_pct": 15.0,
        "windows": results,
    }


# ---------------------------------------------------------
# 10. ACTUAL VS FORECAST FOR ONE ARIMAX BACKTEST
# ---------------------------------------------------------
# Return the eight weekly predictions for one particular
# site and backtest window.
#
# Example:
#
# /model-evaluation/arimax/site/SITE_003/window/1
#
# The dashboard will use these points to plot:
#
# actual demand vs ARIMAX forecast.
@router.get(
    "/arimax/site/{site_id}/window/{window}"
)
def arimax_site_window_predictions(
    site_id: str,
    window: int,
):

    if window not in range(1, 6):
        raise HTTPException(
            status_code=400,
            detail=(
                "Window must be between "
                "1 and 5."
            ),
        )

    try:
        df = pd.read_csv(
            ARIMAX_PREDICTIONS_PATH
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=(
                "ARIMAX prediction report "
                "not found."
            ),
        )

    site_id = (
        site_id
        .strip()
        .upper()
    )

    result = df[
        (
            df["site_id"]
            == site_id
        )
        & (
            df["window"]
            == window
        )
    ].copy()

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No predictions found for "
                f"{site_id}, window {window}."
            ),
        )

    result = result.sort_values(
        "week_start"
    )

    points = []

    for _, row in result.iterrows():

        points.append(
            {
                "week_start": str(
                    row["week_start"]
                ),
                "actual": round(
                    float(
                        row["consumed_tonnes"]
                    ),
                    2,
                ),
                "forecast": round(
                    float(
                        row["forecast"]
                    ),
                    2,
                ),
                "planned_pour": round(
                    float(
                        row[
                            "planned_pour_tonnes"
                        ]
                    ),
                    2,
                ),
            }
        )

    return {
        "model": "ARIMAX",
        "site_id": site_id,
        "window": window,
        "points": points,
    }
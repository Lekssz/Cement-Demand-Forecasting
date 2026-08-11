from pathlib import Path
import os

import numpy as np
import pandas as pd
import mlflow

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

from arimax import (
    load_data,
    create_weekly_data,
    calculate_mape
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Keep exactly the same model settings used in the
# successful ARIMAX experiment.
FORECAST_HORIZON = 8
ARIMA_ORDER = (0, 1, 1)

# Test the model across five different historical
# 8-week forecasting periods.
N_BACKTEST_WINDOWS = 5


# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------
# 1. EVALUATE ONE BACKTEST WINDOW
# ---------------------------------------------------------
def evaluate_window(
    weekly_df,
    test_start,
    window_number
):

    predictions_all_sites = []

    print(
        f"\nRunning backtest window {window_number}..."
    )


    # -----------------------------------------------------
    # TRAIN ONE MODEL FOR EACH SITE
    # -----------------------------------------------------
    for site_id, site_data in weekly_df.groupby("site_id"):

        site_data = (
            site_data
            .sort_values("week_start")
            .reset_index(drop=True)
        )


        # -------------------------------------------------
        # TRAIN / TEST SPLIT
        # -------------------------------------------------
        # Everything before test_start is historical data.
        train_data = site_data.iloc[:test_start].copy()

        # The following 8 weeks form this backtest window.
        test_data = (
            site_data
            .iloc[
                test_start:
                test_start + FORECAST_HORIZON
            ]
            .copy()
        )


        # -------------------------------------------------
        # PREPARE TARGET
        # -------------------------------------------------
        train_target = (
            train_data["consumed_tonnes"]
            .astype(float)
            .reset_index(drop=True)
        )


        # -------------------------------------------------
        # PREPARE PLANNED POUR FEATURE
        # -------------------------------------------------
        train_exog = (
            train_data[["planned_pour_tonnes"]]
            .astype(float)
            .reset_index(drop=True)
        )

        test_exog = (
            test_data[["planned_pour_tonnes"]]
            .astype(float)
            .reset_index(drop=True)
        )


        # -------------------------------------------------
        # TRAIN ARIMAX
        # -------------------------------------------------
        model = SARIMAX(
            endog=train_target,
            exog=train_exog,
            order=ARIMA_ORDER,
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted_model = model.fit(
            disp=False
        )


        # -------------------------------------------------
        # FORECAST NEXT 8 WEEKS
        # -------------------------------------------------
        predictions = fitted_model.forecast(
            steps=FORECAST_HORIZON,
            exog=test_exog
        )

        test_data["forecast"] = np.asarray(
            predictions
        )

        test_data["window"] = window_number

        predictions_all_sites.append(
            test_data
        )


    return pd.concat(
        predictions_all_sites,
        ignore_index=True
    )


# ---------------------------------------------------------
# 2. CALCULATE METRICS
# ---------------------------------------------------------
def evaluate_predictions(predictions_df):

    actual = predictions_df["consumed_tonnes"]
    predicted = predictions_df["forecast"]

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    mape = calculate_mape(
        actual,
        predicted
    )

    return mae, rmse, mape


# ---------------------------------------------------------
# 3. RUN ROLLING BACKTEST
# ---------------------------------------------------------
def run_backtest(weekly_df):

    all_windows = []
    window_metrics = []

    # Every site currently has the same number of
    # complete weekly observations.
    weeks_per_site = (
        weekly_df
        .groupby("site_id")
        .size()
        .min()
    )


    # -----------------------------------------------------
    # FIND FIRST BACKTEST WINDOW
    # -----------------------------------------------------
    # With 156 weeks and five 8-week windows:
    #
    # Window 1 trains on first 116 weeks
    # Window 2 trains on first 124 weeks
    # Window 3 trains on first 132 weeks
    # Window 4 trains on first 140 weeks
    # Window 5 trains on first 148 weeks
    #
    # Each window then forecasts the following 8 weeks.
    first_test_start = (
        weeks_per_site
        - (
            N_BACKTEST_WINDOWS
            * FORECAST_HORIZON
        )
    )


    for window_number in range(
        1,
        N_BACKTEST_WINDOWS + 1
    ):

        test_start = (
            first_test_start
            + (
                (window_number - 1)
                * FORECAST_HORIZON
            )
        )


        # Run this historical forecast window.
        window_predictions = evaluate_window(
            weekly_df=weekly_df,
            test_start=test_start,
            window_number=window_number
        )


        # Calculate performance for this window.
        mae, rmse, mape = evaluate_predictions(
            window_predictions
        )


        # Get the actual dates represented by the window.
        window_start_date = (
            window_predictions["week_start"].min()
        )

        window_end_date = (
            window_predictions["week_start"].max()
        )


        window_metrics.append({
            "window": window_number,
            "start_date": window_start_date,
            "end_date": window_end_date,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        })


        all_windows.append(
            window_predictions
        )


        print(
            f"Window {window_number} completed | "
            f"MAPE: {mape:.2f}%"
        )


    predictions_df = pd.concat(
        all_windows,
        ignore_index=True
    )

    metrics_df = pd.DataFrame(
        window_metrics
    )

    return predictions_df, metrics_df


# ---------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------
def main():

    # -----------------------------------------------------
    # LOAD WEEKLY DATA
    # -----------------------------------------------------
    df = load_data()

    weekly_df = create_weekly_data(df)

    print("\nARIMAX Rolling Backtest")
    print("=" * 50)

    print(
        f"Sites: "
        f"{weekly_df['site_id'].nunique()}"
    )

    print(
        f"Backtest windows: "
        f"{N_BACKTEST_WINDOWS}"
    )

    print(
        f"Forecast horizon: "
        f"{FORECAST_HORIZON} weeks"
    )


    # -----------------------------------------------------
    # MLFLOW CONFIGURATION
    # -----------------------------------------------------
    # Use an AWS/shared tracking URI later if one is set.
    # Otherwise use our current local MLflow database.
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        "cement-demand-forecasting"
    )


    # -----------------------------------------------------
    # START MLFLOW RUN
    # -----------------------------------------------------
    with mlflow.start_run(
        run_name="ARIMAX_rolling_backtest"
    ):

        mlflow.log_params({
            "model": "ARIMAX",
            "arima_order": str(ARIMA_ORDER),
            "external_feature": "planned_pour_tonnes",
            "forecast_horizon_weeks": FORECAST_HORIZON,
            "backtest_windows": N_BACKTEST_WINDOWS,
            "number_of_sites": weekly_df["site_id"].nunique()
        })


        # -------------------------------------------------
        # RUN BACKTEST
        # -------------------------------------------------
        predictions_df, metrics_df = run_backtest(
            weekly_df
        )


        # -------------------------------------------------
        # CALCULATE SUMMARY PERFORMANCE
        # -------------------------------------------------
        mean_mae = metrics_df["MAE"].mean()
        mean_rmse = metrics_df["RMSE"].mean()
        mean_mape = metrics_df["MAPE"].mean()

        std_mape = metrics_df["MAPE"].std()


        # Overall error across every prediction generated
        # during all five backtest windows.
        overall_mae, overall_rmse, overall_mape = (
            evaluate_predictions(
                predictions_df
            )
        )


        # -------------------------------------------------
        # LOG SUMMARY METRICS TO MLFLOW
        # -------------------------------------------------
        mlflow.log_metrics({
            "mean_backtest_MAE": mean_mae,
            "mean_backtest_RMSE": mean_rmse,
            "mean_backtest_MAPE": mean_mape,
            "std_backtest_MAPE": std_mape,
            "overall_backtest_MAE": overall_mae,
            "overall_backtest_RMSE": overall_rmse,
            "overall_backtest_MAPE": overall_mape
        })


        # -------------------------------------------------
        # SAVE BACKTEST RESULTS
        # -------------------------------------------------
        results_path = (
            PROJECT_ROOT
            / "reports"
            / "arimax_backtest_results.csv"
        )

        metrics_df.to_csv(
            results_path,
            index=False
        )

        # Store the results table with the MLflow run.
        mlflow.log_artifact(
            str(results_path)
        )


    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------
    print("\n" + "=" * 70)
    print("ARIMAX BACKTEST RESULTS")
    print("=" * 70)

    print(
        metrics_df.to_string(
            index=False
        )
    )


    print("\nBacktest Summary:")
    print(
        f"Mean MAPE:    "
        f"{mean_mape:.2f}%"
    )

    print(
        f"MAPE Std Dev: "
        f"{std_mape:.2f}%"
    )

    print(
        f"Overall MAPE: "
        f"{overall_mape:.2f}%"
    )

    print(
        "\nProject target: MAPE <= 15%"
    )


# ---------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
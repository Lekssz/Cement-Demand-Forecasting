from pathlib import Path
import os

import numpy as np
import pandas as pd
import mlflow

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
FORECAST_HORIZON = 8
ARIMA_ORDER = (1, 1, 1)
N_BACKTEST_WINDOWS = 5


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "operations_cleaned.csv"
)


# ---------------------------------------------------------
# 1. LOAD CLEANED DATA
# ---------------------------------------------------------
def load_data():

    print("\nLoading cleaned cement data...")

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    # Sorting is important because we later use the first
    # and last inventory observations inside each week.
    df = df.sort_values(
        ["site_id", "date"]
    )

    return df


# ---------------------------------------------------------
# 2. CREATE WEEKLY MODELLING DATA
# ---------------------------------------------------------
def create_weekly_data(df):

    # Create Monday as the beginning of each week.
    df["week_start"] = (
        df["date"]
        - pd.to_timedelta(
            df["date"].dt.weekday,
            unit="D"
        )
    )

    weekly_df = (
        df.groupby(
            ["site_id", "week_start"],
            as_index=False
        )
        .agg(

            # Forecast target.
            consumed_tonnes=(
                "consumed_tonnes",
                "sum"
            ),

            # Known future construction schedule.
            planned_pour_tonnes=(
                "planned_pour_tonnes",
                "sum"
            ),

            # Inventory at the beginning of the week.
            opening_inventory_tonnes=(
                "opening_inventory_tonnes",
                "first"
            ),

            # Inventory at the end of the week.
            closing_inventory_tonnes=(
                "closing_inventory_tonnes",
                "last"
            ),

            # Total deliveries during the week.
            deliveries_tonnes=(
                "deliveries_tonnes",
                "sum"
            ),

            # Site silo capacity.
            silo_capacity=(
                "silo_capacity",
                "last"
            ),

            days_in_week=(
                "date",
                "nunique"
            )
        )
    )

    # Remove incomplete weeks.
    weekly_df = weekly_df[
        weekly_df["days_in_week"] == 7
    ].copy()

    weekly_df = weekly_df.sort_values(
        ["site_id", "week_start"]
    ).reset_index(drop=True)

    return weekly_df


# ---------------------------------------------------------
# 3. CREATE SAFE HISTORICAL INVENTORY FEATURES
# ---------------------------------------------------------
def create_inventory_features(weekly_df):

    feature_frames = []

    for site_id, site_data in weekly_df.groupby("site_id"):

        site_data = (
            site_data
            .sort_values("week_start")
            .copy()
        )

        # -------------------------------------------------
        # PREVIOUS CLOSING INVENTORY
        # -------------------------------------------------
        # For week T, use only inventory from week T-1.
        site_data["previous_closing_inventory"] = (
            site_data["closing_inventory_tonnes"]
            .shift(1)
        )


        # -------------------------------------------------
        # PREVIOUS WEEK DELIVERIES
        # -------------------------------------------------
        # Again, only historical deliveries are used.
        site_data["previous_week_deliveries"] = (
            site_data["deliveries_tonnes"]
            .shift(1)
        )


        # -------------------------------------------------
        # PREVIOUS INVENTORY UTILISATION
        # -------------------------------------------------
        # How full was the silo before the target week?
        site_data["previous_inventory_utilisation"] = (
            site_data["closing_inventory_tonnes"]
            .shift(1)
            /
            site_data["silo_capacity"]
            .shift(1)
        )


        # -------------------------------------------------
        # RECENT 4-WEEK INVENTORY LEVEL
        # -------------------------------------------------
        # Average closing inventory from the four weeks
        # BEFORE the week being predicted.
        site_data["avg_4week_inventory"] = (
            site_data["closing_inventory_tonnes"]
            .shift(1)
            .rolling(4)
            .mean()
        )


        # -------------------------------------------------
        # RECENT 4-WEEK DELIVERY LEVEL
        # -------------------------------------------------
        site_data["avg_4week_deliveries"] = (
            site_data["deliveries_tonnes"]
            .shift(1)
            .rolling(4)
            .mean()
        )

        feature_frames.append(
            site_data
        )


    return pd.concat(
        feature_frames,
        ignore_index=True
    )


# ---------------------------------------------------------
# 4. MAPE FUNCTION
# ---------------------------------------------------------
def calculate_mape(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mask = actual != 0

    return (
        np.mean(
            np.abs(
                (actual[mask] - predicted[mask])
                / actual[mask]
            )
        )
        * 100
    )


# ---------------------------------------------------------
# 5. RUN ONE BACKTEST WINDOW
# ---------------------------------------------------------
def evaluate_window(
    weekly_df,
    test_start,
    window_number
):

    all_predictions = []

    print(
        f"\nRunning inventory backtest "
        f"window {window_number}..."
    )


    for site_id, site_data in weekly_df.groupby("site_id"):

        site_data = (
            site_data
            .sort_values("week_start")
            .reset_index(drop=True)
        )


        # -------------------------------------------------
        # TRAIN / TEST SPLIT
        # -------------------------------------------------
        train_data = (
            site_data
            .iloc[:test_start]
            .copy()
        )

        test_data = (
            site_data
            .iloc[
                test_start:
                test_start + FORECAST_HORIZON
            ]
            .copy()
        )


        # -------------------------------------------------
        # REMOVE EARLY ROWS WITHOUT LAG FEATURES
        # -------------------------------------------------
        inventory_features = [
            "previous_closing_inventory",
            "previous_week_deliveries",
            "previous_inventory_utilisation",
            "avg_4week_inventory",
            "avg_4week_deliveries"
        ]

        train_data = train_data.dropna(
            subset=inventory_features
        )


        # -------------------------------------------------
        # TARGET
        # -------------------------------------------------
        train_target = (
            train_data["consumed_tonnes"]
            .astype(float)
            .reset_index(drop=True)
        )


        # -------------------------------------------------
        # TRAINING FEATURES
        # -------------------------------------------------
        # Historical planned pours plus historical inventory
        # information known before each training week.
        train_exog = (
            train_data[[
                "planned_pour_tonnes",
                "previous_closing_inventory",
                "previous_week_deliveries",
                "previous_inventory_utilisation",
                "avg_4week_inventory",
                "avg_4week_deliveries"
            ]]
            .astype(float)
            .reset_index(drop=True)
        )


        # -------------------------------------------------
        # INVENTORY KNOWN AT FORECAST ORIGIN
        # -------------------------------------------------
        # This is the crucial leakage protection.
        #
        # We ONLY use inventory information from the final
        # training week. We do NOT use actual inventory or
        # deliveries from any of the future test weeks.

        original_train = (
            site_data
            .iloc[:test_start]
            .copy()
        )

        origin_closing_inventory = (
            original_train[
                "closing_inventory_tonnes"
            ]
            .iloc[-1]
        )

        origin_previous_delivery = (
            original_train[
                "deliveries_tonnes"
            ]
            .iloc[-1]
        )

        origin_silo_capacity = (
            original_train[
                "silo_capacity"
            ]
            .iloc[-1]
        )

        origin_inventory_utilisation = (
            origin_closing_inventory
            / origin_silo_capacity
        )

        origin_avg_inventory = (
            original_train[
                "closing_inventory_tonnes"
            ]
            .tail(4)
            .mean()
        )

        origin_avg_deliveries = (
            original_train[
                "deliveries_tonnes"
            ]
            .tail(4)
            .mean()
        )


        # -------------------------------------------------
        # FUTURE FEATURES
        # -------------------------------------------------
        # Planned pours can vary because we assume MIG knows
        # the next 8-week pour schedule.
        #
        # Inventory values are frozen at the forecast origin
        # because future realised inventory is unknown.

        test_exog = pd.DataFrame({

            "planned_pour_tonnes":
                test_data[
                    "planned_pour_tonnes"
                ].astype(float).values,

            "previous_closing_inventory":
                origin_closing_inventory,

            "previous_week_deliveries":
                origin_previous_delivery,

            "previous_inventory_utilisation":
                origin_inventory_utilisation,

            "avg_4week_inventory":
                origin_avg_inventory,

            "avg_4week_deliveries":
                origin_avg_deliveries
        })


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

        all_predictions.append(
            test_data
        )


    return pd.concat(
        all_predictions,
        ignore_index=True
    )


# ---------------------------------------------------------
# 6. CALCULATE PERFORMANCE
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
# 7. RUN FIVE BACKTEST WINDOWS
# ---------------------------------------------------------
def run_backtest(weekly_df):

    all_windows = []
    window_metrics = []

    weeks_per_site = (
        weekly_df
        .groupby("site_id")
        .size()
        .min()
    )

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


        window_predictions = evaluate_window(
            weekly_df=weekly_df,
            test_start=test_start,
            window_number=window_number
        )


        mae, rmse, mape = evaluate_predictions(
            window_predictions
        )


        window_metrics.append({

            "window": window_number,

            "start_date":
                window_predictions[
                    "week_start"
                ].min(),

            "end_date":
                window_predictions[
                    "week_start"
                ].max(),

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
# 8. MAIN
# ---------------------------------------------------------
def main():

    df = load_data()

    weekly_df = create_weekly_data(df)

    weekly_df = create_inventory_features(
        weekly_df
    )


    print("\nARIMAX + Inventory Rolling Backtest")
    print("=" * 60)

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
    # If the team later provides an AWS MLflow tracking URI,
    # this environment variable can point to it without
    # changing the model code.

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
    # TRACK EXPERIMENT
    # -----------------------------------------------------
    with mlflow.start_run(
        run_name="ARIMAX_planned_pours_inventory"
    ):

        mlflow.log_params({

            "model": "ARIMAX",

            "arima_order":
                str(ARIMA_ORDER),

            "forecast_horizon_weeks":
                FORECAST_HORIZON,

            "backtest_windows":
                N_BACKTEST_WINDOWS,

            "features":
                (
                    "planned_pour_tonnes,"
                    "previous_closing_inventory,"
                    "previous_week_deliveries,"
                    "previous_inventory_utilisation,"
                    "avg_4week_inventory,"
                    "avg_4week_deliveries"
                )
        })


        predictions_df, metrics_df = (
            run_backtest(
                weekly_df
            )
        )


        # -------------------------------------------------
        # SUMMARY METRICS
        # -------------------------------------------------
        mean_mae = (
            metrics_df["MAE"].mean()
        )

        mean_rmse = (
            metrics_df["RMSE"].mean()
        )

        mean_mape = (
            metrics_df["MAPE"].mean()
        )

        std_mape = (
            metrics_df["MAPE"].std()
        )


        overall_mae, overall_rmse, overall_mape = (
            evaluate_predictions(
                predictions_df
            )
        )


        # -------------------------------------------------
        # LOG TO MLFLOW
        # -------------------------------------------------
        mlflow.log_metrics({

            "mean_backtest_MAE":
                mean_mae,

            "mean_backtest_RMSE":
                mean_rmse,

            "mean_backtest_MAPE":
                mean_mape,

            "std_backtest_MAPE":
                std_mape,

            "overall_backtest_MAPE":
                overall_mape
        })


        # -------------------------------------------------
        # SAVE RESULTS
        # -------------------------------------------------
        results_path = (
            PROJECT_ROOT
            / "reports"
            / "arimax_inventory_backtest_results.csv"
        )

        metrics_df.to_csv(
            results_path,
            index=False
        )

        mlflow.log_artifact(
            str(results_path)
        )


    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------
    print("\n" + "=" * 70)

    print(
        "ARIMAX + INVENTORY BACKTEST RESULTS"
    )

    print("=" * 70)

    print(
        metrics_df.to_string(
            index=False
        )
    )


    print("\nBacktest Summary:")

    print(
        f"Mean MAPE:     "
        f"{mean_mape:.2f}%"
    )

    print(
        f"MAPE Std Dev:  "
        f"{std_mape:.2f}%"
    )

    print(
        f"Overall MAPE:  "
        f"{overall_mape:.2f}%"
    )

    print(
        "\nCurrent planned-pours-only "
        "backtest MAPE: 11.42%"
    )

    print(
        "Project target: MAPE <= 15%"
    )


# ---------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
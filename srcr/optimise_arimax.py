from itertools import product
from pathlib import Path
import os

import pandas as pd
import mlflow

import backtest_arimax as backtest
from arimax import load_data, create_weekly_data


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Values we want to test for ARIMAX (p, d, q).
P_VALUES = [0, 1, 2]
D_VALUES = [0, 1]
Q_VALUES = [0, 1, 2]

FORECAST_HORIZON = 8

# Windows 1-4 are used for tuning.
# Window 5 stays untouched for final testing.
VALIDATION_WINDOWS = 4


# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------
# 1. TEST ONE ARIMAX ORDER
# ---------------------------------------------------------
def test_order(weekly_df, order):

    # backtest_arimax.py already uses this global setting.
    # Change it temporarily for each experiment.
    backtest.ARIMA_ORDER = order

    weeks_per_site = (
        weekly_df
        .groupby("site_id")
        .size()
        .min()
    )

    # We have five 8-week periods in total.
    # The first four are validation periods.
    first_test_start = (
        weeks_per_site
        - (5 * FORECAST_HORIZON)
    )

    mapes = []


    # -----------------------------------------------------
    # RUN VALIDATION WINDOWS 1-4
    # -----------------------------------------------------
    for window_number in range(
        1,
        VALIDATION_WINDOWS + 1
    ):

        test_start = (
            first_test_start
            + (
                (window_number - 1)
                * FORECAST_HORIZON
            )
        )

        predictions = backtest.evaluate_window(
            weekly_df=weekly_df,
            test_start=test_start,
            window_number=window_number
        )

        _, _, mape = backtest.evaluate_predictions(
            predictions
        )

        mapes.append(mape)


    return sum(mapes) / len(mapes)


# ---------------------------------------------------------
# 2. MAIN GRID SEARCH
# ---------------------------------------------------------
def main():

    df = load_data()

    weekly_df = create_weekly_data(df)


    # -----------------------------------------------------
    # MLFLOW CONFIGURATION
    # -----------------------------------------------------
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(
        "cement-demand-forecasting"
    )


    # -----------------------------------------------------
    # CREATE ARIMAX COMBINATIONS
    # -----------------------------------------------------
    orders = list(
        product(
            P_VALUES,
            D_VALUES,
            Q_VALUES
        )
    )

    results = []

    print("\nARIMAX Grid Search")
    print("=" * 50)

    print(
        f"Testing {len(orders)} configurations..."
    )


    # -----------------------------------------------------
    # TEST EVERY ORDER
    # -----------------------------------------------------
    for number, order in enumerate(
        orders,
        start=1
    ):

        print(
            f"\n[{number}/{len(orders)}] "
            f"Testing ARIMAX{order}"
        )

        try:

            mean_mape = test_order(
                weekly_df,
                order
            )

            results.append({
                "order": order,
                "mean_validation_MAPE": mean_mape
            })


            # Record each experiment in MLflow.
            with mlflow.start_run(
                run_name=f"ARIMAX_{order}"
            ):

                mlflow.log_param(
                    "order",
                    str(order)
                )

                mlflow.log_param(
                    "feature",
                    "planned_pour_tonnes"
                )

                mlflow.log_metric(
                    "mean_validation_MAPE",
                    mean_mape
                )


            print(
                f"Mean MAPE: "
                f"{mean_mape:.2f}%"
            )


        except Exception as error:

            print(
                f"Failed: {error}"
            )


    # -----------------------------------------------------
    # FIND BEST MODEL
    # -----------------------------------------------------
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "mean_validation_MAPE"
    )

    print("\n" + "=" * 50)
    print("TUNING RESULTS")
    print("=" * 50)

    print(
        results_df.to_string(
            index=False
        )
    )


    best_order = results_df.iloc[0]["order"]
    best_mape = results_df.iloc[0]["mean_validation_MAPE"]


    print("\nBest ARIMAX configuration:")

    print(
        f"Order: {best_order}"
    )

    print(
        f"Validation MAPE: "
        f"{best_mape:.2f}%"
    )


    # -----------------------------------------------------
    # FINAL UNTOUCHED WINDOW 5
    # -----------------------------------------------------
    backtest.ARIMA_ORDER = best_order

    weeks_per_site = (
        weekly_df
        .groupby("site_id")
        .size()
        .min()
    )

    final_predictions = backtest.evaluate_window(
        weekly_df=weekly_df,
        test_start=weeks_per_site - FORECAST_HORIZON,
        window_number=5
    )

    final_mae, final_rmse, final_mape = (
        backtest.evaluate_predictions(
            final_predictions
        )
    )


    print("\nFinal Window 5 Test:")
    print(
        f"MAE:  {final_mae:.2f} tonnes"
    )

    print(
        f"RMSE: {final_rmse:.2f} tonnes"
    )

    print(
        f"MAPE: {final_mape:.2f}%"
    )


# ---------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
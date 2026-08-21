from pathlib import Path

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# The business requirement is to forecast cement demand
# up to 8 weeks ahead.
FORECAST_HORIZON = 8

# First ARIMA configuration to experiment with.
# We are NOT claiming that (1,1,1) is the best setting yet.
ARIMA_ORDER = (1, 1, 1)


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------
# Find the project root automatically.
# This makes the script less dependent on where it is run from.
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

    return df


# ---------------------------------------------------------
# 2. CREATE WEEKLY CEMENT DEMAND
# ---------------------------------------------------------
def create_weekly_data(df):

    # Assign every daily observation to the Monday
    # representing the beginning of its week.
    df["week_start"] = (
        df["date"]
        - pd.to_timedelta(
            df["date"].dt.weekday,
            unit="D"
        )
    )

    # Sum daily cement consumption into weekly demand
    # for every construction site.
    weekly_df = (
        df.groupby(
            ["site_id", "week_start"],
            as_index=False
        )
        .agg(
            consumed_tonnes=("consumed_tonnes", "sum"),
            days_in_week=("date", "nunique")
        )
    )

    # Remove incomplete weeks because partial weeks could
    # make weekly demand appear artificially low.
    weekly_df = weekly_df[
        weekly_df["days_in_week"] == 7
    ].copy()

    weekly_df = weekly_df.sort_values(
        ["site_id", "week_start"]
    )

    return weekly_df


# ---------------------------------------------------------
# 3. MAPE FUNCTION
# ---------------------------------------------------------
def calculate_mape(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    # Avoid dividing by zero if a zero-demand week exists.
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
# 4. TRAIN ARIMA FOR EVERY SITE
# ---------------------------------------------------------
def run_arima(weekly_df):

    all_forecasts = []

    print(
        f"\nRunning ARIMA{ARIMA_ORDER} "
        f"for {weekly_df['site_id'].nunique()} sites..."
    )

    for site_id, site_data in weekly_df.groupby("site_id"):

        site_data = (
            site_data
            .sort_values("week_start")
            .copy()
        )

        # -------------------------------------------------
        # TRAIN / TEST SPLIT
        # -------------------------------------------------
        # Keep the final 8 weeks completely unseen by
        # the model so we can evaluate an 8-week forecast.
        train_data = site_data.iloc[:-FORECAST_HORIZON]

        test_data = (
            site_data
            .iloc[-FORECAST_HORIZON:]
            .copy()
        )

        # -------------------------------------------------
        # TRAIN ARIMA
        # -------------------------------------------------
        model = ARIMA(
            train_data["consumed_tonnes"],
            order=ARIMA_ORDER
        )

        fitted_model = model.fit()

        # -------------------------------------------------
        # FORECAST NEXT 8 WEEKS
        # -------------------------------------------------
        predictions = fitted_model.forecast(
            steps=FORECAST_HORIZON
        )

        test_data["arima_forecast"] = np.asarray(
            predictions
        )

        all_forecasts.append(test_data)

        print(f"Completed {site_id}")

    return pd.concat(
        all_forecasts,
        ignore_index=True
    )


# ---------------------------------------------------------
# 5. CALCULATE OVERALL ARIMA PERFORMANCE
# ---------------------------------------------------------
def evaluate_overall(arima_df):

    actual = arima_df["consumed_tonnes"]
    predicted = arima_df["arima_forecast"]

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
# 6. CALCULATE PERFORMANCE FOR EACH SITE
# ---------------------------------------------------------
def evaluate_by_site(arima_df):

    results = []

    for site_id, site_data in arima_df.groupby("site_id"):

        actual = site_data["consumed_tonnes"]
        predicted = site_data["arima_forecast"]

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

        results.append({
            "site_id": site_id,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------
# 7. RUN COMPLETE ARIMA EXPERIMENT
# ---------------------------------------------------------
def main():

    # Load original cleaned daily data.
    df = load_data()

    # Convert daily consumption into complete weekly periods.
    weekly_df = create_weekly_data(df)

    print("\nWeekly modelling dataset:")
    print(f"Rows: {len(weekly_df)}")
    print(
        f"Sites: "
        f"{weekly_df['site_id'].nunique()}"
    )

    print(
        f"Date range: "
        f"{weekly_df['week_start'].min()} "
        f"to "
        f"{weekly_df['week_start'].max()}"
    )

    # Train and forecast every site.
    arima_df = run_arima(
        weekly_df
    )

    # Calculate overall model performance.
    mae, rmse, mape = evaluate_overall(
        arima_df
    )

    # Calculate individual site performance.
    site_performance = evaluate_by_site(
        arima_df
    )


    # -----------------------------------------------------
    # DISPLAY OVERALL RESULTS
    # -----------------------------------------------------
    print("\n" + "=" * 50)
    print("ARIMA OVERALL PERFORMANCE")
    print("=" * 50)

    print(f"Model: ARIMA{ARIMA_ORDER}")
    print(f"Forecast horizon: {FORECAST_HORIZON} weeks")
    print(f"MAE:  {mae:.2f} tonnes")
    print(f"RMSE: {rmse:.2f} tonnes")
    print(f"MAPE: {mape:.2f}%")


    # -----------------------------------------------------
    # DISPLAY SITE-LEVEL RESULTS
    # -----------------------------------------------------
    print("\nARIMA Performance by Site:")

    print(
        site_performance
        .sort_values("MAPE")
        .to_string(index=False)
    )


# ---------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
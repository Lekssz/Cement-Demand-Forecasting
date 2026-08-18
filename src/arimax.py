from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Use the same settings as our ARIMA experiment so that
# the comparison remains fair.
FORECAST_HORIZON = 8
ARIMA_ORDER = (1, 1, 1)

#ARIMA_ORDER = (0, 1, 1) #changed after optimisation
# ---------------------------------------------------------
# FILE PATH
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

    return df


# ---------------------------------------------------------
# 2. CREATE WEEKLY MODELLING DATA
# ---------------------------------------------------------
def create_weekly_data(df):

    # Create the Monday representing the beginning
    # of each weekly period.
    df["week_start"] = (
        df["date"]
        - pd.to_timedelta(
            df["date"].dt.weekday,
            unit="D"
        )
    )

    # Aggregate both actual cement consumption and
    # planned cement pours into weekly values.
    weekly_df = (
        df.groupby(
            ["site_id", "week_start"],
            as_index=False
        )
        .agg(
            consumed_tonnes=(
                "consumed_tonnes",
                "sum"
            ),

            planned_pour_tonnes=(
                "planned_pour_tonnes",
                "sum"
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

    return weekly_df.sort_values(
        ["site_id", "week_start"]
    )


# ---------------------------------------------------------
# 3. MAPE FUNCTION
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
# 4. RUN ARIMAX FOR ALL SITES
# ---------------------------------------------------------
def run_arimax(weekly_df):

    results = []

    print(
        f"\nRunning ARIMAX{ARIMA_ORDER} "
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
        train_data = site_data.iloc[:-FORECAST_HORIZON]

        test_data = (
            site_data
            .iloc[-FORECAST_HORIZON:]
            .copy()
        )


        # -------------------------------------------------
        # PREPARE TARGET DATA
        # -------------------------------------------------
        train_target = (
            train_data["consumed_tonnes"]
            .astype(float)
            .reset_index(drop=True)
        )


        # -------------------------------------------------
        # PREPARE PLANNED POUR FEATURE
        # -------------------------------------------------
        # Training exogenous values are historical
        # planned pour schedules.
        train_exog = (
            train_data[["planned_pour_tonnes"]]
            .astype(float)
            .reset_index(drop=True)
        )

        # Test exogenous values represent the planned
        # pour schedule available for the next 8 weeks.
        test_exog = (
            test_data[["planned_pour_tonnes"]]
            .astype(float)
            .reset_index(drop=True)
        )


        # -------------------------------------------------
        # TRAIN ARIMAX MODEL
        # -------------------------------------------------
        model = SARIMAX(
            endog=train_target,
            exog=train_exog,
            order=ARIMA_ORDER,

            # No seasonal component because our
            # seasonality analysis found weak evidence.
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

        test_data["arimax_forecast"] = np.asarray(
            predictions
        )

        results.append(test_data)

        print(f"Completed {site_id}")


    return pd.concat(
        results,
        ignore_index=True
    )


# ---------------------------------------------------------
# 5. CALCULATE OVERALL PERFORMANCE
# ---------------------------------------------------------
def evaluate_overall(arimax_df):

    actual = arimax_df["consumed_tonnes"]
    predicted = arimax_df["arimax_forecast"]

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
# 6. CALCULATE PERFORMANCE BY SITE
# ---------------------------------------------------------
def evaluate_by_site(arimax_df):

    site_results = []

    for site_id, site_data in arimax_df.groupby("site_id"):

        actual = site_data["consumed_tonnes"]
        predicted = site_data["arimax_forecast"]

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

        site_results.append({
            "site_id": site_id,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        })

    return pd.DataFrame(site_results)


# ---------------------------------------------------------
# 7. RUN COMPLETE ARIMAX EXPERIMENT
# ---------------------------------------------------------
def main():

    df = load_data()

    weekly_df = create_weekly_data(df)

    print("\nWeekly modelling dataset:")
    print(f"Rows: {len(weekly_df)}")
    print(
        f"Sites: "
        f"{weekly_df['site_id'].nunique()}"
    )
 # -----------------------------------------------------
    # MLFLOW CONFIGURATION
    # -----------------------------------------------------
    # Store MLflow experiment information in a local
    # SQLite database inside this project.
    mlflow.set_tracking_uri(
        f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )

    # Create or use the cement forecasting experiment.
    mlflow.set_experiment(
        "cement-demand-forecasting"
    )


    # -----------------------------------------------------
    # RUN AND TRACK ARIMAX EXPERIMENT
    # -----------------------------------------------------
    with mlflow.start_run(
        run_name="ARIMAX_planned_pours"
    ):

        # Record the settings used for this experiment.
        mlflow.log_params({
            "model": "ARIMAX",
            "arima_order": str(ARIMA_ORDER),
            "forecast_horizon_weeks": FORECAST_HORIZON,
            "external_feature": "planned_pour_tonnes",
            "number_of_sites": weekly_df["site_id"].nunique()
        })   

    # Run model.
    arimax_df = run_arimax(
        weekly_df
    )


    # Evaluate overall performance.
    mae, rmse, mape = evaluate_overall(
        arimax_df
    )


    # Evaluate each site.
    site_performance = evaluate_by_site(
        arimax_df
    )


    # -----------------------------------------------------
    # DISPLAY OVERALL RESULTS
    # -----------------------------------------------------
    print("\n" + "=" * 50)
    print("ARIMAX OVERALL PERFORMANCE")
    print("=" * 50)

    print(f"Model: ARIMAX{ARIMA_ORDER}")
    print("External feature: planned_pour_tonnes")
    print(f"Forecast horizon: {FORECAST_HORIZON} weeks")

    print(f"MAE:  {mae:.2f} tonnes")
    print(f"RMSE: {rmse:.2f} tonnes")
    print(f"MAPE: {mape:.2f}%")


    # -----------------------------------------------------
    # DISPLAY PERFORMANCE BY SITE
    # -----------------------------------------------------
    print("\nARIMAX Performance by Site:")

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
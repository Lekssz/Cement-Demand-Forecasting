import pandas as pd


# ---------------------------------------------------------
# 1. LOAD CLEANED DATA
# ---------------------------------------------------------
# Load the cleaned cement operations dataset produced during
# the data cleaning stage of the project.
df = pd.read_csv("data/processed/operations_cleaned.csv")


# ---------------------------------------------------------
# 2. PREPARE DATE COLUMN
# ---------------------------------------------------------
# Convert the date column into datetime format so we can
# perform time-series operations.
df["date"] = pd.to_datetime(df["date"])


# ---------------------------------------------------------
# 3. CREATE WEEKLY TIME PERIOD
# ---------------------------------------------------------
# Assign every daily record to the Monday of its week.
# This allows us to aggregate daily cement consumption
# into weekly demand.
df["week_start"] = (
    df["date"]
    - pd.to_timedelta(df["date"].dt.weekday, unit="D")
)


# ---------------------------------------------------------
# 4. CALCULATE WEEKLY CEMENT CONSUMPTION PER SITE
# ---------------------------------------------------------
# Sum daily cement consumption for each site and week.
# We also count how many days are present in each week so
# that incomplete weeks can be identified.
weekly_df = (
    df.groupby(["site_id", "week_start"], as_index=False)
      .agg(
          consumed_tonnes=("consumed_tonnes", "sum"),
          days_in_week=("date", "nunique")
      )
)


# ---------------------------------------------------------
# 5. REMOVE INCOMPLETE WEEKS
# ---------------------------------------------------------
# Keep only complete 7-day periods so partial weeks do not
# artificially reduce weekly cement demand.
weekly_df = weekly_df[
    weekly_df["days_in_week"] == 7
].copy()


# ---------------------------------------------------------
# 6. VERIFY WEEKLY DATASET
# ---------------------------------------------------------
print("\nWeekly cement consumption:")
print(weekly_df.head(15))

print("\nDataset shape:")
print(weekly_df.shape)

print("\nNumber of sites:")
print(weekly_df["site_id"].nunique())

print("\nDate range:")
print("Start:", weekly_df["week_start"].min())
print("End:", weekly_df["week_start"].max())


# ---------------------------------------------------------
# 7. CREATE 8-WEEK BASELINE FORECAST
# ---------------------------------------------------------
# The project requires forecasts up to 8 weeks ahead.
# Therefore, the final 8 weeks for every site are reserved
# as unseen test data.
#
# Our simple baseline assumes that future weekly demand will
# remain equal to the most recently observed weekly demand.

baseline_results = []

for site_id, site_data in weekly_df.groupby("site_id"):

    # Ensure observations are in chronological order.
    site_data = site_data.sort_values("week_start").copy()

    # Reserve the final 8 weeks for testing.
    train_data = site_data.iloc[:-8]
    test_data = site_data.iloc[-8:].copy()

    # Most recent demand known at the time the forecast is made.
    last_known_demand = train_data["consumed_tonnes"].iloc[-1]

    # Use the same value as the baseline forecast for all
    # eight future weeks.
    test_data["baseline_forecast"] = last_known_demand

    baseline_results.append(test_data)


# Combine the test forecasts from all sites.
baseline_df = pd.concat(
    baseline_results,
    ignore_index=True
)


# ---------------------------------------------------------
# 8. VERIFY BASELINE FORECAST
# ---------------------------------------------------------
print("\nBaseline forecast sample:")
print(
    baseline_df[
        [
            "site_id",
            "week_start",
            "consumed_tonnes",
            "baseline_forecast"
        ]
    ].head(16)
)

print("\nNumber of baseline predictions:")
print(len(baseline_df))

# ---------------------------------------------------------
# 9. EVALUATE BASELINE PERFORMANCE
# ---------------------------------------------------------
# The baseline provides a benchmark that future forecasting
# models must outperform.
#
# MAE  = average absolute forecasting error in tonnes.
# RMSE = similar to MAE but penalises larger errors more heavily.
# MAPE = average percentage forecasting error. The project
#        target requires MAPE to be 15% or lower.

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


actual = baseline_df["consumed_tonnes"]
predicted = baseline_df["baseline_forecast"]


# Mean Absolute Error
mae = mean_absolute_error(actual, predicted)


# Root Mean Squared Error
rmse = np.sqrt(
    mean_squared_error(actual, predicted)
)


# Check that actual demand does not contain zero values
# before calculating MAPE.
zero_values = (actual == 0).sum()

if zero_values == 0:

    mape = np.mean(
        np.abs((actual - predicted) / actual)
    ) * 100

    print("\nBaseline Performance:")
    print(f"MAE:  {mae:.2f} tonnes")
    print(f"RMSE: {rmse:.2f} tonnes")
    print(f"MAPE: {mape:.2f}%")

else:

    print("\nBaseline Performance:")
    print(f"MAE:  {mae:.2f} tonnes")
    print(f"RMSE: {rmse:.2f} tonnes")
    print(
        f"MAPE cannot be calculated normally because "
        f"{zero_values} actual observations contain zero demand."
    )

    # ---------------------------------------------------------
# 10. EVALUATE BASELINE PERFORMANCE FOR EACH SITE
# ---------------------------------------------------------
# The overall MAPE tells us how the baseline performs across
# all sites combined. We also calculate MAPE separately for
# each site so we can identify sites that are easier or
# harder to forecast.

site_performance = []

for site_id, site_data in baseline_df.groupby("site_id"):

    actual = site_data["consumed_tonnes"]
    predicted = site_data["baseline_forecast"]

    # Only calculate MAPE if the site has no zero-demand weeks.
    if (actual == 0).sum() == 0:

        site_mape = (
            np.mean(
                np.abs((actual - predicted) / actual)
            ) * 100
        )

        site_mae = mean_absolute_error(
            actual,
            predicted
        )

        site_rmse = np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        )

        site_performance.append({
            "site_id": site_id,
            "MAE": site_mae,
            "RMSE": site_rmse,
            "MAPE": site_mape
        })


# Convert the results into a dataframe.
site_performance_df = pd.DataFrame(site_performance)


# Display performance for all sites.
print("\nBaseline Performance by Site:")
print(
    site_performance_df
    .sort_values("MAPE")
    .to_string(index=False)
)
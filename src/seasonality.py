import pandas as pd

from arima_model import load_data, create_weekly_data


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Keep the final 8 weeks unseen because these are our test
# weeks for forecasting evaluation.
FORECAST_HORIZON = 8

# Examine repeating patterns between 2 and 60 weeks.
MAX_LAG = 60


# ---------------------------------------------------------
# 1. LOAD AND PREPARE WEEKLY DATA
# ---------------------------------------------------------
df = load_data()

weekly_df = create_weekly_data(df)


# ---------------------------------------------------------
# 2. CHECK SEASONAL AUTOCORRELATION
# ---------------------------------------------------------
# Autocorrelation asks:
#
# "How similar is demand now to demand X weeks ago?"
#
# Example:
# lag 4  = compare with 4 weeks ago
# lag 13 = compare with 13 weeks ago
# lag 52 = compare with roughly one year ago
#
# We only use the TRAINING period here. The final 8 weeks
# remain untouched.

lag_results = []

for lag in range(2, MAX_LAG + 1):

    site_correlations = []

    for site_id, site_data in weekly_df.groupby("site_id"):

        site_data = site_data.sort_values("week_start")

        # Exclude the final 8 test weeks.
        train_data = site_data.iloc[:-FORECAST_HORIZON]

        demand = train_data["consumed_tonnes"]

        # First difference helps remove general trends so
        # repeating patterns are easier to identify.
        demand_diff = demand.diff().dropna()

        correlation = demand_diff.autocorr(lag=lag)

        if pd.notna(correlation):
            site_correlations.append(correlation)

    # Average the correlation across all construction sites.
    mean_correlation = sum(site_correlations) / len(site_correlations)

    lag_results.append({
        "lag_weeks": lag,
        "mean_autocorrelation": mean_correlation
    })


# ---------------------------------------------------------
# 3. DISPLAY STRONGEST REPEATING WEEKLY PATTERNS
# ---------------------------------------------------------
lag_df = pd.DataFrame(lag_results)

strongest_lags = (
    lag_df
    .sort_values(
        "mean_autocorrelation",
        ascending=False
    )
    .head(10)
)

print("\nStrongest weekly seasonal lags:")
print(
    strongest_lags.to_string(index=False)
)
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
# We are testing the Random Forest on exactly the same
# forecasting problem as ARIMAX:
#
#     forecast the next 8 weeks of cement consumption.
#
# We keep the teammate's Random Forest settings so that
# we are testing their current model idea fairly.
FORECAST_HORIZON = 8

N_ESTIMATORS = 300
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "operations_cleaned.csv"
)

ARIMAX_RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "arimax_backtest_results.csv"
)

RF_RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "rf_backtest_results.csv"
)

RF_SITE_RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "rf_site_backtest_results.csv"
)

RF_SITE_SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "rf_site_summary.csv"
)

RF_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "rf_backtest_predictions.csv"
)


# ---------------------------------------------------------
# 2. RANDOM FOREST FEATURES
# ---------------------------------------------------------
# These are the same main feature ideas used in the
# teammate's existing Random Forest pipeline.
#
# Historical consumption:
#   - 1 week ago
#   - 2 weeks ago
#   - 4 weeks ago
#   - 8 weeks ago
#
# Historical averages:
#   - previous 4 weeks
#   - previous 8 weeks
#
# Other information:
#   - planned pours
#   - rainfall
#   - temperature
#   - silo capacity
#
# Site characteristics:
#   - behaviour
#   - cement type
#   - region
#   - site ID
NUMERIC_FEATURES = [
    "consumed_tonnes_lag_1",
    "consumed_tonnes_lag_2",
    "consumed_tonnes_lag_4",
    "consumed_tonnes_lag_8",
    "consumed_tonnes_rollmean_4",
    "consumed_tonnes_rollmean_8",
    "planned_pour_tonnes",
    "rain_mm",
    "avg_temp_c",
    "silo_capacity",
]

CATEGORICAL_FEATURES = [
    "behavior",
    "cement_type",
    "region",
    "site_id",
]

FEATURE_COLS = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

TARGET_COLS = [
    f"consumed_tonnes_t_plus_{h}"
    for h in range(
        1,
        FORECAST_HORIZON + 1
    )
]


# ---------------------------------------------------------
# 3. LOAD CLEANED DATA
# ---------------------------------------------------------
def load_data():

    print("\nLoading cleaned cement data...")

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


# ---------------------------------------------------------
# 4. CREATE WEEKLY DATA
# ---------------------------------------------------------
# IMPORTANT:
#
# ARIMAX used Monday as the beginning of each week.
#
# The existing Random Forest pipeline uses pandas "W",
# which creates Sunday-ending weeks.
#
# For this comparison we use the ARIMAX weekly definition
# so both models are evaluated on the exact same periods.
def create_weekly_data(df):

    df = df.copy()

    # Monday representing the start of the week.
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
            consumed_tonnes=(
                "consumed_tonnes",
                "sum"
            ),

            planned_pour_tonnes=(
                "planned_pour_tonnes",
                "sum"
            ),

            rain_mm=(
                "rain_mm",
                "mean"
            ),

            avg_temp_c=(
                "avg_temp_c",
                "mean"
            ),

            silo_capacity=(
                "silo_capacity",
                "last"
            ),

            behavior=(
                "behavior",
                "last"
            ),

            cement_type=(
                "cement_type",
                "last"
            ),

            region=(
                "region",
                "last"
            ),

            days_in_week=(
                "date",
                "nunique"
            )
        )
    )

    # Keep only complete 7-day weeks.
    #
    # This matches what we did during ARIMAX modelling
    # and prevents partial first/last weeks from affecting
    # the comparison.
    weekly_df = weekly_df[
        weekly_df["days_in_week"] == 7
    ].copy()

    weekly_df = weekly_df.sort_values(
        ["site_id", "week_start"]
    ).reset_index(drop=True)

    return weekly_df


# ---------------------------------------------------------
# 5. CREATE LAG FEATURES
# ---------------------------------------------------------
def add_lag_features(weekly_df):

    df = weekly_df.copy()

    df = df.sort_values(
        ["site_id", "week_start"]
    )

    # -----------------------------------------------------
    # CONSUMPTION LAGS
    # -----------------------------------------------------
    for lag in [1, 2, 4, 8]:

        df[
            f"consumed_tonnes_lag_{lag}"
        ] = (
            df.groupby("site_id")[
                "consumed_tonnes"
            ]
            .shift(lag)
        )


    # -----------------------------------------------------
    # GROUP-SAFE ROLLING MEANS
    # -----------------------------------------------------
    # We first shift by one week because today's
    # consumption must not be used to predict the future.
    #
    # transform() keeps the rolling calculation completely
    # separate for every site.
    #
    # This also avoids rolling values accidentally crossing
    # from SITE_001 into SITE_002, etc.
    for window in [4, 8]:

        df[
            f"consumed_tonnes_rollmean_{window}"
        ] = (
            df.groupby("site_id")[
                "consumed_tonnes"
            ]
            .transform(
                lambda series:
                series
                .shift(1)
                .rolling(
                    window=window,
                    min_periods=window
                )
                .mean()
            )
        )

    return df


# ---------------------------------------------------------
# 6. CREATE 8 FUTURE TARGETS
# ---------------------------------------------------------
# One Random Forest input row predicts:
#
#     t+1
#     t+2
#     ...
#     t+8
#
# simultaneously.
def add_future_targets(df):

    df = df.copy()

    for h in range(
        1,
        FORECAST_HORIZON + 1
    ):

        df[
            f"consumed_tonnes_t_plus_{h}"
        ] = (
            df.groupby("site_id")[
                "consumed_tonnes"
            ]
            .shift(-h)
        )

    return df


# ---------------------------------------------------------
# 7. BUILD MODELLING DATASET
# ---------------------------------------------------------
def build_model_data(df):

    weekly_df = create_weekly_data(df)

    model_df = add_lag_features(
        weekly_df
    )

    model_df = add_future_targets(
        model_df
    )

    # Remove rows that do not yet have sufficient
    # historical lag information or enough future
    # observations to create all eight targets.
    required_cols = (
        FEATURE_COLS
        + TARGET_COLS
    )

    model_df = (
        model_df
        .dropna(
            subset=required_cols
        )
        .copy()
    )

    print("\nWeekly modelling dataset:")
    print(
        f"Rows:  {len(model_df)}"
    )
    print(
        f"Sites: "
        f"{model_df['site_id'].nunique()}"
    )

    return weekly_df, model_df


# ---------------------------------------------------------
# 8. BUILD RANDOM FOREST
# ---------------------------------------------------------
def build_random_forest():

    # Convert text/categorical features into numerical
    # values using one-hot encoding.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES
            ),

            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES
            ),
        ]
    )

    random_forest = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    model = Pipeline(
        steps=[
            (
                "preprocess",
                preprocessor
            ),
            (
                "random_forest",
                random_forest
            ),
        ]
    )

    return model


# ---------------------------------------------------------
# 9. SAFE MAPE
# ---------------------------------------------------------
# This is intentionally the same style of MAPE used in
# our ARIMAX experiments.
#
# Zero-demand observations are excluded because dividing
# by zero would make MAPE invalid.
def calculate_mape(actual, predicted):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    mask = actual != 0

    if mask.sum() == 0:
        return np.nan

    return (
        np.mean(
            np.abs(
                (
                    actual[mask]
                    - predicted[mask]
                )
                / actual[mask]
            )
        )
        * 100
    )


# ---------------------------------------------------------
# 10. CALCULATE METRICS
# ---------------------------------------------------------
def calculate_metrics(
    actual,
    predicted
):

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
# 11. GET SAME WINDOWS AS ARIMAX
# ---------------------------------------------------------
def get_backtest_windows(
    weekly_df
):

    # -----------------------------------------------------
    # BEST OPTION:
    # Read the exact historical dates already used by
    # ARIMAX.
    # -----------------------------------------------------
    if ARIMAX_RESULTS_PATH.exists():

        arimax_results = pd.read_csv(
            ARIMAX_RESULTS_PATH
        )

        arimax_results[
            "start_date"
        ] = pd.to_datetime(
            arimax_results[
                "start_date"
            ]
        )

        arimax_results[
            "end_date"
        ] = pd.to_datetime(
            arimax_results[
                "end_date"
            ]
        )

        windows = []

        for _, row in (
            arimax_results
            .sort_values("window")
            .iterrows()
        ):

            windows.append({
                "window": int(
                    row["window"]
                ),

                "start_date":
                    row["start_date"],

                "end_date":
                    row["end_date"],
            })

        return windows


    # -----------------------------------------------------
    # FALLBACK:
    # Reconstruct five non-overlapping 8-week windows
    # from the final 40 complete weeks.
    # -----------------------------------------------------
    dates = (
        weekly_df[
            "week_start"
        ]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    total_backtest_weeks = (
        5
        * FORECAST_HORIZON
    )

    backtest_dates = dates.iloc[
        -total_backtest_weeks:
    ]

    windows = []

    for window_number in range(
        1,
        6
    ):

        start_index = (
            (window_number - 1)
            * FORECAST_HORIZON
        )

        window_dates = (
            backtest_dates.iloc[
                start_index:
                start_index
                + FORECAST_HORIZON
            ]
        )

        windows.append({
            "window":
                window_number,

            "start_date":
                window_dates.iloc[0],

            "end_date":
                window_dates.iloc[-1],
        })

    return windows


# ---------------------------------------------------------
# 12. TRAIN AND FORECAST ONE WINDOW
# ---------------------------------------------------------
def evaluate_window(
    weekly_df,
    model_df,
    window_number,
    test_start,
    test_end
):

    print(
        f"\nRunning RF backtest "
        f"window {window_number}..."
    )

    print(
        f"Forecast period: "
        f"{test_start.date()} "
        f"to {test_end.date()}"
    )


    # -----------------------------------------------------
    # FORECAST ORIGIN
    # -----------------------------------------------------
    # If the first forecast week is 25 March, the model
    # receives the feature row from the week immediately
    # before it.
    origin_date = (
        test_start
        - pd.Timedelta(
            weeks=1
        )
    )


    # -----------------------------------------------------
    # PREVENT TARGET LEAKAGE
    # -----------------------------------------------------
    # A training row predicts eight future weeks.
    #
    # Therefore it can only be included if its t+8 target
    # occurs BEFORE the start of this test window.
    #
    # Example:
    #
    # Training row:
    # 29 Jan
    #
    # t+8 target:
    # 25 Mar
    #
    # If 25 Mar is our test start, that training row must
    # NOT be used because it already contains the answer
    # from the test period.
    model_df = model_df.copy()

    model_df[
        "target_end_date"
    ] = (
        model_df["week_start"]
        + pd.to_timedelta(
            FORECAST_HORIZON,
            unit="W"
        )
    )

    train_df = model_df[
        model_df[
            "target_end_date"
        ] < test_start
    ].copy()


    # -----------------------------------------------------
    # FEATURE ROW USED TO MAKE FORECAST
    # -----------------------------------------------------
    origin_df = model_df[
        model_df[
            "week_start"
        ] == origin_date
    ].copy()

    expected_sites = (
        weekly_df[
            "site_id"
        ]
        .nunique()
    )

    if (
        origin_df[
            "site_id"
        ]
        .nunique()
        != expected_sites
    ):
        raise ValueError(
            f"Window {window_number}: "
            f"expected {expected_sites} "
            f"sites at forecast origin "
            f"{origin_date.date()}, "
            f"but found "
            f"{origin_df['site_id'].nunique()}."
        )


    # -----------------------------------------------------
    # TRAIN GLOBAL RANDOM FOREST
    # -----------------------------------------------------
    X_train = train_df[
        FEATURE_COLS
    ]

    y_train = train_df[
        TARGET_COLS
    ]

    X_origin = origin_df[
        FEATURE_COLS
    ]

    print(
        f"Training rows: "
        f"{len(X_train)}"
    )

    model = build_random_forest()

    model.fit(
        X_train,
        y_train
    )


    # -----------------------------------------------------
    # PREDICT ALL 8 WEEKS
    # -----------------------------------------------------
    predictions = model.predict(
        X_origin
    )


    # -----------------------------------------------------
    # CONVERT MATRIX INTO FORECAST TABLE
    # -----------------------------------------------------
    forecast_rows = []

    origin_df = (
        origin_df
        .reset_index(drop=True)
    )

    for row_index in range(
        len(origin_df)
    ):

        site_id = origin_df.loc[
            row_index,
            "site_id"
        ]

        for h in range(
            1,
            FORECAST_HORIZON + 1
        ):

            forecast_date = (
                origin_date
                + pd.Timedelta(
                    weeks=h
                )
            )

            predicted_value = (
                predictions[
                    row_index,
                    h - 1
                ]
            )

            forecast_rows.append({
                "window":
                    window_number,

                "site_id":
                    site_id,

                "week_start":
                    forecast_date,

                "forecast":
                    predicted_value,
            })

    forecast_df = pd.DataFrame(
        forecast_rows
    )


    # -----------------------------------------------------
    # GET ACTUAL DEMAND
    # -----------------------------------------------------
    actual_df = weekly_df[
        (
            weekly_df[
                "week_start"
            ] >= test_start
        )
        &
        (
            weekly_df[
                "week_start"
            ] <= test_end
        )
    ][
        [
            "site_id",
            "week_start",
            "consumed_tonnes",
        ]
    ].copy()


    # -----------------------------------------------------
    # JOIN FORECAST TO ACTUAL VALUES
    # -----------------------------------------------------
    result_df = forecast_df.merge(
        actual_df,
        on=[
            "site_id",
            "week_start"
        ],
        how="inner"
    )

    expected_rows = (
        expected_sites
        * FORECAST_HORIZON
    )

    if len(result_df) != expected_rows:

        raise ValueError(
            f"Window {window_number}: "
            f"expected {expected_rows} "
            f"predictions but got "
            f"{len(result_df)}."
        )

    return result_df


# ---------------------------------------------------------
# 13. SITE PERFORMANCE FOR ONE WINDOW
# ---------------------------------------------------------
def evaluate_sites(
    predictions_df
):

    site_results = []

    for site_id, site_data in (
        predictions_df
        .groupby("site_id")
    ):

        mae, rmse, mape = (
            calculate_metrics(
                site_data[
                    "consumed_tonnes"
                ],
                site_data[
                    "forecast"
                ]
            )
        )

        site_results.append({
            "window":
                int(
                    site_data[
                        "window"
                    ].iloc[0]
                ),

            "site_id":
                site_id,

            "MAE":
                mae,

            "RMSE":
                rmse,

            "MAPE":
                mape,
        })

    return pd.DataFrame(
        site_results
    )


# ---------------------------------------------------------
# 14. RUN ALL FIVE WINDOWS
# ---------------------------------------------------------
def run_backtest(
    weekly_df,
    model_df
):

    windows = get_backtest_windows(
        weekly_df
    )

    all_predictions = []

    window_metrics = []

    site_metrics = []


    for window in windows:

        predictions = evaluate_window(
            weekly_df=weekly_df,
            model_df=model_df,
            window_number=window[
                "window"
            ],
            test_start=window[
                "start_date"
            ],
            test_end=window[
                "end_date"
            ]
        )

        mae, rmse, mape = (
            calculate_metrics(
                predictions[
                    "consumed_tonnes"
                ],
                predictions[
                    "forecast"
                ]
            )
        )

        window_metrics.append({
            "window":
                window["window"],

            "start_date":
                window[
                    "start_date"
                ],

            "end_date":
                window[
                    "end_date"
                ],

            "MAE":
                mae,

            "RMSE":
                rmse,

            "MAPE":
                mape,
        })

        window_site_metrics = (
            evaluate_sites(
                predictions
            )
        )

        site_metrics.append(
            window_site_metrics
        )

        all_predictions.append(
            predictions
        )

        print(
            f"Window "
            f"{window['window']} "
            f"completed | "
            f"MAPE: "
            f"{mape:.2f}%"
        )


    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    metrics_df = pd.DataFrame(
        window_metrics
    )

    site_metrics_df = pd.concat(
        site_metrics,
        ignore_index=True
    )

    return (
        predictions_df,
        metrics_df,
        site_metrics_df
    )


# ---------------------------------------------------------
# 15. CREATE SITE SUMMARY ACROSS ALL 5 WINDOWS
# ---------------------------------------------------------
def create_site_summary(
    predictions_df
):

    site_summary = []

    for site_id, site_data in (
        predictions_df
        .groupby("site_id")
    ):

        mae, rmse, mape = (
            calculate_metrics(
                site_data[
                    "consumed_tonnes"
                ],
                site_data[
                    "forecast"
                ]
            )
        )

        site_summary.append({
            "site_id":
                site_id,

            "MAE":
                mae,

            "RMSE":
                rmse,

            "MAPE":
                mape,

            "meets_15_percent_target":
                mape <= 15,
        })

    return (
        pd.DataFrame(
            site_summary
        )
        .sort_values("MAPE")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------
# 16. COMPARE RANDOM FOREST WITH ARIMAX
# ---------------------------------------------------------
def compare_with_arimax(
    rf_results
):

    if not ARIMAX_RESULTS_PATH.exists():

        print(
            "\nARIMAX results file "
            "was not found."
        )

        return


    arimax = pd.read_csv(
        ARIMAX_RESULTS_PATH
    )

    comparison = (
        arimax[
            [
                "window",
                "MAPE"
            ]
        ]
        .rename(
            columns={
                "MAPE":
                    "ARIMAX_MAPE"
            }
        )
        .merge(
            rf_results[
                [
                    "window",
                    "MAPE"
                ]
            ].rename(
                columns={
                    "MAPE":
                        "RF_MAPE"
                }
            ),
            on="window"
        )
    )

    comparison[
        "winner"
    ] = np.where(
        comparison[
            "ARIMAX_MAPE"
        ]
        <
        comparison[
            "RF_MAPE"
        ],
        "ARIMAX",
        "Random Forest"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ARIMAX VS RANDOM FOREST"
    )

    print(
        "=" * 70
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "\nAverage ARIMAX MAPE: "
        f"{comparison['ARIMAX_MAPE'].mean():.2f}%"
    )

    print(
        "Average RF MAPE:     "
        f"{comparison['RF_MAPE'].mean():.2f}%"
    )


# ---------------------------------------------------------
# 17. MAIN
# ---------------------------------------------------------
def main():

    df = load_data()

    weekly_df, model_df = (
        build_model_data(
            df
        )
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "RANDOM FOREST "
        "5-WINDOW BACKTEST"
    )

    print(
        "=" * 70
    )

    print(
        f"Forecast horizon: "
        f"{FORECAST_HORIZON} weeks"
    )

    print(
        f"Trees: "
        f"{N_ESTIMATORS}"
    )

    print(
        f"Sites: "
        f"{weekly_df['site_id'].nunique()}"
    )


    # -----------------------------------------------------
    # RUN BACKTEST
    # -----------------------------------------------------
    (
        predictions_df,
        metrics_df,
        site_metrics_df
    ) = run_backtest(
        weekly_df,
        model_df
    )


    # -----------------------------------------------------
    # CREATE SITE SUMMARY
    # -----------------------------------------------------
    site_summary_df = (
        create_site_summary(
            predictions_df
        )
    )


    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------
    RF_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_df.to_csv(
        RF_RESULTS_PATH,
        index=False
    )

    site_metrics_df.to_csv(
        RF_SITE_RESULTS_PATH,
        index=False
    )

    site_summary_df.to_csv(
        RF_SITE_SUMMARY_PATH,
        index=False
    )

    predictions_df.to_csv(
        RF_PREDICTIONS_PATH,
        index=False
    )


    # -----------------------------------------------------
    # DISPLAY WINDOW RESULTS
    # -----------------------------------------------------
    print(
        "\n"
        + "=" * 70
    )

    print(
        "RANDOM FOREST "
        "BACKTEST RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        metrics_df.to_string(
            index=False
        )
    )


    mean_mape = (
        metrics_df[
            "MAPE"
        ]
        .mean()
    )

    std_mape = (
        metrics_df[
            "MAPE"
        ]
        .std()
    )


    print(
        "\nRF Backtest Summary:"
    )

    print(
        f"Mean MAPE:    "
        f"{mean_mape:.2f}%"
    )

    print(
        f"MAPE Std Dev: "
        f"{std_mape:.2f}%"
    )

    print(
        "\nProject target: "
        "MAPE <= 15%"
    )


    # -----------------------------------------------------
    # DISPLAY SITE RESULTS
    # -----------------------------------------------------
    passed_sites = (
        site_summary_df[
            "meets_15_percent_target"
        ]
        .sum()
    )

    total_sites = len(
        site_summary_df
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RANDOM FOREST "
        "SITE PERFORMANCE"
    )

    print(
        "=" * 70
    )

    print(
        site_summary_df.to_string(
            index=False
        )
    )

    print(
        "\nSites <= 15% MAPE: "
        f"{passed_sites}/"
        f"{total_sites}"
    )

    print(
        "Sites > 15% MAPE:  "
        f"{total_sites - passed_sites}/"
        f"{total_sites}"
    )


    # -----------------------------------------------------
    # FINAL ARIMAX COMPARISON
    # -----------------------------------------------------
    compare_with_arimax(
        metrics_df
    )


# ---------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
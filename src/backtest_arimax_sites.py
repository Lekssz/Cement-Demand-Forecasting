from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


# ---------------------------------------------------------
# 1. PROJECT PATH
# ---------------------------------------------------------
# Add the srcr folder so we can reuse the ARIMAX
# backtesting code we already built instead of rewriting it.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "srcr")
)

from backtest_arimax import (
    load_data,
    create_weekly_data,
    run_backtest,
    calculate_mape,
)


# ---------------------------------------------------------
# 2. OUTPUT FILES
# ---------------------------------------------------------
# Save:
#
# 1. Every site's result in every window.
# 2. A summary showing performance across all 5 windows.
SITE_WINDOW_RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "arimax_site_window_results.csv"
)

SITE_SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "arimax_site_summary.csv"
)

ARIMAX_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "arimax_backtest_predictions.csv"
)

# ---------------------------------------------------------
# 3. CALCULATE METRICS
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
# 4. CALCULATE SITE PERFORMANCE PER WINDOW
# ---------------------------------------------------------
# This answers:
#
# How did SITE_001 perform in Window 1?
# How did SITE_001 perform in Window 2?
# ...
# How did SITE_030 perform in Window 5?
def evaluate_site_windows(
    predictions_df
):

    site_window_results = []

    for (
        window,
        site_id
    ), site_data in predictions_df.groupby(
        [
            "window",
            "site_id"
        ]
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

        site_window_results.append({
            "window":
                int(window),

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

    return pd.DataFrame(
        site_window_results
    )


# ---------------------------------------------------------
# 5. CREATE SUMMARY FOR EACH SITE
# ---------------------------------------------------------
# We calculate two useful measures:
#
# overall_MA​PE
#     Error across all 40 forecast weeks
#     (5 windows × 8 weeks).
#
# mean_window_MAPE
#     Average of that site's five individual
#     window MAPE values.
#
# failed_windows
#     How many of the five windows had MAPE > 15%.
#
# This lets us distinguish:
#
# "one unusually bad period"
#
# from
#
# "ARIMAX consistently struggles with this site".
def create_site_summary(
    predictions_df,
    site_window_df
):

    site_summary = []

    for site_id, site_data in (
        predictions_df
        .groupby("site_id")
    ):

        mae, rmse, overall_mape = (
            calculate_metrics(
                site_data[
                    "consumed_tonnes"
                ],
                site_data[
                    "forecast"
                ]
            )
        )

        window_data = (
            site_window_df[
                site_window_df[
                    "site_id"
                ] == site_id
            ]
        )

        mean_window_mape = (
            window_data[
                "MAPE"
            ]
            .mean()
        )

        best_window_mape = (
            window_data[
                "MAPE"
            ]
            .min()
        )

        worst_window_mape = (
            window_data[
                "MAPE"
            ]
            .max()
        )

        failed_windows = (
            window_data[
                "MAPE"
            ]
            .gt(15)
            .sum()
        )

        passed_windows = (
            5
            - failed_windows
        )

        site_summary.append({
            "site_id":
                site_id,

            "MAE":
                mae,

            "RMSE":
                rmse,

            "overall_MAPE":
                overall_mape,

            "mean_window_MAPE":
                mean_window_mape,

            "best_window_MAPE":
                best_window_mape,

            "worst_window_MAPE":
                worst_window_mape,

            "passed_windows":
                passed_windows,

            "failed_windows":
                failed_windows,

            "meets_overall_15_percent_target":
                overall_mape <= 15,
        })

    return (
        pd.DataFrame(
            site_summary
        )
        .sort_values(
            "overall_MAPE"
        )
        .reset_index(
            drop=True
        )
    )


# ---------------------------------------------------------
# 6. CREATE WINDOW TABLE
# ---------------------------------------------------------
# Produce an easy-to-read table:
#
# site       W1    W2    W3    W4    W5
# SITE_001   ...   ...   ...   ...   ...
def create_window_table(
    site_window_df
):

    table = (
        site_window_df
        .pivot(
            index="site_id",
            columns="window",
            values="MAPE"
        )
        .rename(
            columns={
                1: "W1",
                2: "W2",
                3: "W3",
                4: "W4",
                5: "W5",
            }
        )
    )

    return table


# ---------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------
def main():

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------
    df = load_data()

    weekly_df = create_weekly_data(
        df
    )


    # -----------------------------------------------------
    # RUN EXISTING FIVE-WINDOW ARIMAX BACKTEST
    # -----------------------------------------------------
    print(
        "\nRunning ARIMAX "
        "site-level analysis..."
    )

    predictions_df, window_metrics = (
        run_backtest(
            weekly_df
        )
    )


    # -----------------------------------------------------
    # SITE × WINDOW RESULTS
    # -----------------------------------------------------
    site_window_df = (
        evaluate_site_windows(
            predictions_df
        )
    )


    # -----------------------------------------------------
    # OVERALL SITE SUMMARY
    # -----------------------------------------------------
    site_summary_df = (
        create_site_summary(
            predictions_df,
            site_window_df
        )
    )


    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------
    SITE_WINDOW_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    site_window_df.to_csv(
        SITE_WINDOW_RESULTS_PATH,
        index=False
    )

    site_summary_df.to_csv(
        SITE_SUMMARY_PATH,
        index=False
    )
    predictions_df.to_csv(
        ARIMAX_PREDICTIONS_PATH,
        index=False
    )

    # -----------------------------------------------------
    # DISPLAY WINDOW TABLE
    # -----------------------------------------------------
    window_table = (
        create_window_table(
            site_window_df
        )
    )

    print(
        "\n"
        + "=" * 85
    )

    print(
        "ARIMAX SITE MAPE "
        "BY BACKTEST WINDOW"
    )

    print(
        "=" * 85
    )

    print(
        window_table
        .round(2)
        .to_string()
    )


    # -----------------------------------------------------
    # DISPLAY SITE SUMMARY
    # -----------------------------------------------------
    print(
        "\n"
        + "=" * 100
    )

    print(
        "ARIMAX FIVE-WINDOW "
        "SITE SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        site_summary_df
        .round(2)
        .to_string(
            index=False
        )
    )


    # -----------------------------------------------------
    # TARGET SUMMARY
    # -----------------------------------------------------
    passed_sites = (
        site_summary_df[
            "meets_overall_15_percent_target"
        ]
        .sum()
    )

    total_sites = len(
        site_summary_df
    )

    print(
        "\nSUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        f"Sites <= 15% overall MAPE: "
        f"{passed_sites}/"
        f"{total_sites}"
    )

    print(
        f"Sites > 15% overall MAPE:  "
        f"{total_sites - passed_sites}/"
        f"{total_sites}"
    )


    # -----------------------------------------------------
    # CONSISTENTLY WEAK SITES
    # -----------------------------------------------------
    # A site failing several windows is more concerning
    # than a site with only one unusual bad period.
    weak_sites = (
        site_summary_df[
            site_summary_df[
                "failed_windows"
            ] >= 3
        ]
        .sort_values(
            [
                "failed_windows",
                "overall_MAPE"
            ],
            ascending=[
                False,
                False
            ]
        )
    )

    print(
        "\nSites failing at least "
        "3 of the 5 windows:"
    )

    if weak_sites.empty:

        print(
            "None"
        )

    else:

        print(
            weak_sites[
                [
                    "site_id",
                    "overall_MAPE",
                    "mean_window_MAPE",
                    "failed_windows",
                ]
            ]
            .round(2)
            .to_string(
                index=False
            )
        )


# ---------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
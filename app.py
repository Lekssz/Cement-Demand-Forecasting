import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta

# Page config
st.set_page_config(
    page_title="Cement Demand Forecasting",
    page_icon="🏗️",
    layout="wide"
)

# ---------------------------------------------------------
# Load model and data
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained pipeline."""
    model_path = "notebooks/models/cement_demand_rf.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_data():
    """Load historical data with engineered features."""
    data_path = "data/processed/operations_cleaned.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={'ite_id': 'site_id'})
        return df
    return None

@st.cache_data
def prepare_weekly_data(df):
    """Recreate weekly aggregated data with features."""
    if df is None:
        return None

    weekly = (
        df
        .groupby('site_id')
        .resample('W', on='date')
        .agg({
            'consumed_tonnes': 'sum',
            'planned_pour_tonnes': 'sum',
            'rain_mm': 'mean',
            'avg_temp_c': 'mean',
            'behavior': 'last',
            'cement_type': 'last',
            'region': 'last',
            'silo_capacity': 'last'
        })
        .reset_index()
    )

    def add_lag_features(df, group_col, target_col, lags, rolling_windows):
        df = df.copy()
        df = df.sort_values([group_col, 'date'])
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)
        for window in rolling_windows:
            df[f'{target_col}_rollmean_{window}'] = (
                df.groupby(group_col)[target_col]
                  .shift(1)
                  .rolling(window=window, min_periods=1)
                  .mean()
            )
        return df

    weekly = add_lag_features(
        df=weekly,
        group_col='site_id',
        target_col='consumed_tonnes',
        lags=[1, 2, 4, 8],
        rolling_windows=[4, 8]
    )
    weekly_model = weekly.dropna().copy()
    return weekly_model

# Feature columns (must match training)
FEATURE_COLS = [
    'consumed_tonnes_lag_1', 'consumed_tonnes_lag_2',
    'consumed_tonnes_lag_4', 'consumed_tonnes_lag_8',
    'consumed_tonnes_rollmean_4', 'consumed_tonnes_rollmean_8',
    'planned_pour_tonnes', 'rain_mm', 'avg_temp_c', 'silo_capacity',
    'behavior', 'cement_type', 'region', 'site_id'
]

CATEGORICAL_COLS = ['behavior', 'cement_type', 'region', 'site_id']
HORIZON = 8

# ---------------------------------------------------------
# Forecasting functions
# ---------------------------------------------------------
def forecast_from_features(model, features_df, n_weeks=8):
    """Run the model on a single feature row. Returns array of length n_weeks."""
    return model.predict(features_df[FEATURE_COLS])[0]


def forecast_site_future(model, weekly_df, site_id, start_date, horizon,
                         exog_override=None, mode='constant'):
    """
    Forecast beyond the dataset using recursive predictions.
    Allows overriding exogenous values per future week.

    Parameters
    ----------
    exog_override : pd.DataFrame, optional
        Columns: ['date', 'planned_pour_tonnes', 'rain_mm', 'avg_temp_c']
        Must cover the forecasted weeks.
    mode : str
        'constant' -> use last known exogenous values throughout
        'scenario' -> use values from exog_override week by week
    """
    site_data = weekly_df[weekly_df['site_id'] == site_id].copy()
    site_data = site_data.sort_values('date')

    last_known_row = site_data.iloc[[-1]].copy()
    last_known_date = last_known_row['date'].iloc[0]

    if start_date <= last_known_date:
        return forecast_site_8_weeks(model, weekly_df, site_id, start_date)

    # Initialize features from last known row
    current_features = last_known_row[FEATURE_COLS].copy()
    current_date = last_known_date
    forecasts = []

    # Pre-build exog lookup: map date -> {planned_pour, rain, temp}
    if mode == 'scenario' and exog_override is not None and not exog_override.empty:
        exog_lookup = {
            pd.Timestamp(row['date']): {
                'planned_pour_tonnes': row['planned_pour_tonnes'],
                'rain_mm': row['rain_mm'],
                'avg_temp_c': row['avg_temp_c']
            }
            for _, row in exog_override.iterrows()
        }
    else:
        exog_lookup = {}

    for h in range(1, horizon + 1):
        # Apply exogenous override if available for this week
        if current_date + timedelta(weeks=1) in exog_lookup:
            ex = exog_lookup[current_date + timedelta(weeks=1)]
            current_features['planned_pour_tonnes'] = ex['planned_pour_tonnes']
            current_features['rain_mm'] = ex['rain_mm']
            current_features['avg_temp_c'] = ex['avg_temp_c']

        y_pred = model.predict(current_features[FEATURE_COLS])[0]
        next_pred = y_pred[0]

        next_date = current_date + timedelta(weeks=1)
        forecasts.append({
            'site_id': site_id,
            'date': next_date,
            'forecast_consumed_tonnes': next_pred
        })

        # Update lag features for next iteration
        new_row = current_features.copy()
        new_row['consumed_tonnes_lag_8'] = new_row['consumed_tonnes_lag_4']
        new_row['consumed_tonnes_lag_4'] = new_row['consumed_tonnes_lag_2']
        new_row['consumed_tonnes_lag_2'] = new_row['consumed_tonnes_lag_1']
        new_row['consumed_tonnes_lag_1'] = next_pred

        # Update rolling means from available lags
        lag_vals = [
            new_row['consumed_tonnes_lag_1'].iloc[0],
            new_row['consumed_tonnes_lag_2'].iloc[0],
            new_row['consumed_tonnes_lag_4'].iloc[0],
            new_row['consumed_tonnes_lag_8'].iloc[0]
        ]
        lag_vals = [v for v in lag_vals if not pd.isna(v)]
        if len(lag_vals) >= 4:
            new_row['consumed_tonnes_rollmean_4'] = np.mean(lag_vals[:4])
            new_row['consumed_tonnes_rollmean_8'] = np.mean(lag_vals)
        elif len(lag_vals) > 0:
            new_row['consumed_tonnes_rollmean_4'] = np.mean(lag_vals)
            new_row['consumed_tonnes_rollmean_8'] = np.mean(lag_vals)

        current_features = new_row
        current_date = next_date

    return pd.DataFrame(forecasts)


def forecast_site_8_weeks(model, weekly_df, site_id, last_date):
    """Forecast 8 weeks from a date that exists in the data."""
    site_data = weekly_df[weekly_df['site_id'] == site_id].copy()
    site_data = site_data.sort_values('date')

    current_row = site_data[site_data['date'] == last_date]
    if current_row.empty:
        return None

    y_future = forecast_from_features(model, current_row)

    future_dates = pd.date_range(
        start=last_date + timedelta(weeks=1),
        periods=HORIZON,
        freq='W'
    )

    return pd.DataFrame({
        'site_id': site_id,
        'date': future_dates,
        'forecast_consumed_tonnes': y_future
    })


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.title("🏗️ Cement Demand Forecasting")
st.markdown("Predict 8-week cement demand per site using Random Forest model")

model = load_model()
df = load_data()
weekly_model = prepare_weekly_data(df)

if model is None:
    st.error("❌ Model not found. Run the training notebook and save the model.")
    st.stop()
if weekly_model is None:
    st.error("❌ Data not found. Check `data/processed/operations_cleaned.csv`")
    st.stop()

# Sidebar - Site selection
sites = sorted(weekly_model['site_id'].unique())
site_id = st.sidebar.selectbox("Select Site", sites)

site_info = weekly_model[weekly_model['site_id'] == site_id].iloc[-1]
last_date_in_data = weekly_model[weekly_model['site_id'] == site_id]['date'].max()

st.sidebar.markdown("---")
st.sidebar.markdown("**Site Info:**")
st.sidebar.markdown(f"- Region: {site_info['region']}")
st.sidebar.markdown(f"- Cement Type: {site_info['cement_type']}")
st.sidebar.markdown(f"- Behavior: {site_info['behavior']}")
st.sidebar.markdown(f"- Silo Capacity: {site_info['silo_capacity']:.0f} tonnes")
st.sidebar.markdown(f"- Last Date in Data: {last_date_in_data.date()}")

# Forecast options
st.sidebar.markdown("---")
forecast_mode = st.sidebar.radio(
    "Forecast Mode",
    ["From last date in data", "Custom start date (within data)", "Future dates (beyond data)"]
)
horizon = st.sidebar.slider("Forecast Horizon (weeks)", 1, 12, HORIZON)

# ---------------------------------------------------------
# Feature inputs - editable for all training features
# ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Edit Training Features")
st.sidebar.caption("Values override the model's defaults")

# Last known feature values for the site
last_row = weekly_model[weekly_model['site_id'] == site_id].iloc[-1].copy()

# Numeric: lag & rolling features (auto-populated from history, but editable)
with st.sidebar.expander("📊 Lag & Rolling Features", expanded=False):
    lag_1 = st.number_input("consumed_tonnes_lag_1", value=float(last_row['consumed_tonnes_lag_1']),
                            min_value=0.0, step=1.0, key='lag_1')
    lag_2 = st.number_input("consumed_tonnes_lag_2", value=float(last_row['consumed_tonnes_lag_2']),
                            min_value=0.0, step=1.0, key='lag_2')
    lag_4 = st.number_input("consumed_tonnes_lag_4", value=float(last_row['consumed_tonnes_lag_4']),
                            min_value=0.0, step=1.0, key='lag_4')
    lag_8 = st.number_input("consumed_tonnes_lag_8", value=float(last_row['consumed_tonnes_lag_8']),
                            min_value=0.0, step=1.0, key='lag_8')
    roll_4 = st.number_input("consumed_tonnes_rollmean_4", value=float(last_row['consumed_tonnes_rollmean_4']),
                             min_value=0.0, step=1.0, key='roll_4')
    roll_8 = st.number_input("consumed_tonnes_rollmean_8", value=float(last_row['consumed_tonnes_rollmean_8']),
                             min_value=0.0, step=1.0, key='roll_8')

with st.sidebar.expander("🌦️ Exogenous Numeric", expanded=True):
    planned_pour = st.number_input("planned_pour_tonnes", value=float(last_row['planned_pour_tonnes']),
                                   min_value=0.0, step=10.0, key='planned_pour')
    rain_mm = st.number_input("rain_mm", value=float(last_row['rain_mm']),
                              min_value=0.0, step=1.0, key='rain_mm')
    avg_temp = st.number_input("avg_temp_c", value=float(last_row['avg_temp_c']),
                               step=0.5, key='avg_temp')
    silo_capacity = st.number_input("silo_capacity", value=float(last_row['silo_capacity']),
                                    min_value=0.0, step=10.0, key='silo_capacity')

with st.sidebar.expander("🏷️ Categorical Features", expanded=False):
    behavior = st.selectbox("behavior", sorted(weekly_model['behavior'].unique()),
                             index=list(sorted(weekly_model['behavior'].unique())).index(last_row['behavior']),
                             key='behavior')
    cement_type = st.selectbox("cement_type", sorted(weekly_model['cement_type'].unique()),
                               index=list(sorted(weekly_model['cement_type'].unique())).index(last_row['cement_type']),
                               key='cement_type')
    region = st.selectbox("region", sorted(weekly_model['region'].unique()),
                          index=list(sorted(weekly_model['region'].unique())).index(last_row['region']),
                          key='region')

# Build a single feature row from the inputs
def build_features_row():
    return pd.DataFrame([{
        'consumed_tonnes_lag_1': lag_1,
        'consumed_tonnes_lag_2': lag_2,
        'consumed_tonnes_lag_4': lag_4,
        'consumed_tonnes_lag_8': lag_8,
        'consumed_tonnes_rollmean_4': roll_4,
        'consumed_tonnes_rollmean_8': roll_8,
        'planned_pour_tonnes': planned_pour,
        'rain_mm': rain_mm,
        'avg_temp_c': avg_temp,
        'silo_capacity': silo_capacity,
        'behavior': behavior,
        'cement_type': cement_type,
        'region': region,
        'site_id': site_id
    }])

# ---------------------------------------------------------
# Main content
# ---------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Forecast Results")

    if forecast_mode == "From last date in data":
        start_date = last_date_in_data
        features_row = build_features_row()
        y_future = forecast_from_features(model, features_row)
        future_dates = pd.date_range(
            start=start_date + timedelta(weeks=1),
            periods=horizon,
            freq='W'
        )
        forecast_df = pd.DataFrame({
            'site_id': site_id,
            'date': future_dates,
            'forecast_consumed_tonnes': y_future[:horizon]
        })

    elif forecast_mode == "Custom start date (within data)":
        available_dates = weekly_model[weekly_model['site_id'] == site_id]['date'].unique()
        start_date = st.sidebar.selectbox("Start Date", available_dates)
        row = weekly_model[
            (weekly_model['site_id'] == site_id) & (weekly_model['date'] == start_date)
        ].iloc[[-1]].copy()
        # Override exogenous inputs with sidebar values
        row['planned_pour_tonnes'] = planned_pour
        row['rain_mm'] = rain_mm
        row['avg_temp_c'] = avg_temp
        row['silo_capacity'] = silo_capacity
        row['behavior'] = behavior
        row['cement_type'] = cement_type
        row['region'] = region
        # Override lags/rolling from sidebar inputs
        row['consumed_tonnes_lag_1'] = lag_1
        row['consumed_tonnes_lag_2'] = lag_2
        row['consumed_tonnes_lag_4'] = lag_4
        row['consumed_tonnes_lag_8'] = lag_8
        row['consumed_tonnes_rollmean_4'] = roll_4
        row['consumed_tonnes_rollmean_8'] = roll_8
        y_future = forecast_from_features(model, row)
        future_dates = pd.date_range(
            start=start_date + timedelta(weeks=1),
            periods=horizon,
            freq='W'
        )
        forecast_df = pd.DataFrame({
            'site_id': site_id,
            'date': future_dates,
            'forecast_consumed_tonnes': y_future[:horizon]
        })

    else:  # Future dates beyond data
        start_date = st.sidebar.date_input(
            "Forecast Start Date",
            value=last_date_in_data.date() + timedelta(weeks=1),
            min_value=last_date_in_data.date() + timedelta(weeks=1)
        )
        start_date = pd.Timestamp(start_date)

        # Per-week exogenous scenario
        future_weeks = pd.date_range(start=start_date, periods=horizon, freq='W')
        scenario_df = pd.DataFrame({
            'date': future_weeks,
            'planned_pour_tonnes': planned_pour,
            'rain_mm': rain_mm,
            'avg_temp_c': avg_temp
        })

        st.markdown("**Per-week exogenous scenario** (applied to all forecast weeks):")
        scenario_df_display = scenario_df.copy()
        scenario_df_display['date'] = scenario_df_display['date'].dt.date
        st.dataframe(scenario_df_display, use_container_width=True, hide_index=True)

        forecast_df = forecast_site_future(
            model=model,
            weekly_df=weekly_model,
            site_id=site_id,
            start_date=start_date,
            horizon=horizon,
            exog_override=scenario_df,
            mode='scenario'
        )

    if forecast_df is not None and not forecast_df.empty:
        display_df = forecast_df.copy()
        display_df['date'] = display_df['date'].dt.date
        display_df['forecast_consumed_tonnes'] = display_df['forecast_consumed_tonnes'].round(2)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.line_chart(
            data=forecast_df.set_index('date')['forecast_consumed_tonnes'],
            use_container_width=True
        )

        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast as CSV",
            data=csv,
            file_name=f"forecast_{site_id}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No forecast generated. Check the selected date.")

with col2:
    st.subheader("📊 Historical Context")
    site_history = weekly_model[weekly_model['site_id'] == site_id].copy()
    site_history = site_history.sort_values('date').tail(8)

    if not site_history.empty:
        hist_display = site_history[['date', 'consumed_tonnes', 'planned_pour_tonnes',
                                     'rain_mm', 'avg_temp_c']].copy()
        hist_display['date'] = hist_display['date'].dt.date
        hist_display['consumed_tonnes'] = hist_display['consumed_tonnes'].round(2)
        st.dataframe(hist_display, use_container_width=True, hide_index=True)

        st.line_chart(
            data=site_history.set_index('date')['consumed_tonnes'],
            use_container_width=True
        )
        st.caption("Last 8 weeks actual consumption")

# ---------------------------------------------------------
# Show current feature inputs being used
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔧 Feature Row Used for Forecasting")
features_used = build_features_row()
st.dataframe(features_used, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.caption("Cement Demand Forecasting Model | Random Forest | 8-week horizon")

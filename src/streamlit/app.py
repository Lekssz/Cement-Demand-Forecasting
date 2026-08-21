"""Streamlit dashboard for the FastAPI inference service.

This app talks to the same API as app1.py (GET /health, GET /sites,
GET /sites/{site_id}/info, POST /forecast) but adds the extra UI
features from the standalone app.py:

  1. Dropdowns (instead of free text) for behavior / cement_type / region
  2. A "Historical Context" panel showing recent actual consumption
  3. A selectbox of in-data dates for "Custom start date" mode

These three features need data the current API does not return yet.
Each one is written to *optionally* use richer data if the API provides
it, and to gracefully fall back to app1's simpler behavior otherwise.

Expected (optional) API additions to unlock full functionality later:

  GET /sites/{site_id}/info  -> may additionally include:
      "categorical_options": {
          "behavior": [...unique values...],
          "cement_type": [...unique values...],
          "region": [...unique values...]
      }
      "available_dates": ["YYYY-MM-DD", ...]   # in-data dates for this site

  GET /sites/{site_id}/history  (new, optional endpoint) ->
      {
        "history": [
          {"date": "YYYY-MM-DD", "consumed_tonnes": ..., "planned_pour_tonnes": ...,
           "rain_mm": ..., "avg_temp_c": ...},
          ...
        ]
      }

Until those exist, the app behaves exactly like app1.py for these three
features (free-text categorical inputs, a plain date picker, no history
panel) but the code is ready to light up automatically once the API
starts returning the extra fields.
"""
import os
from datetime import timedelta

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000").rstrip("/")
DEFAULT_HORIZON = 8

st.set_page_config(page_title="Cement Demand Forecasting", page_icon="🏗️", layout="wide")
st.title("🏗️ Cement Demand Forecasting")
st.caption("Predict cement demand with the shared Random Forest inference service")


def api_request(method, path, **kwargs):
    response = requests.request(method, f"{API_URL}{path}", timeout=30, **kwargs)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=60)
def get_health():
    try:
        return api_request("GET", "/health")
    except requests.RequestException:
        return {"status": "down", "model_loaded": False}


@st.cache_data(ttl=300)
def get_sites():
    return api_request("GET", "/sites")


@st.cache_data(ttl=300)
def get_site_info(site_id):
    return api_request("GET", f"/sites/{site_id}/info")


@st.cache_data(ttl=300)
def get_site_history(site_id):
    """Optional endpoint. Returns None if the API doesn't support it yet."""
    try:
        data = api_request("GET", f"/sites/{site_id}/history")
        return pd.DataFrame(data["history"])
    except (requests.RequestException, KeyError, ValueError):
        return None


health = get_health()
if health.get("status") == "down":
    msg = f"Could not reach the forecasting API at {API_URL}."
    st.error(msg)
    # Give the user a way to clear the cached failure without restarting
    # the container (e.g. when the API finished starting after this page).
    if st.button("Retry API connection"):
        st.cache_data.clear()
        st.rerun()
    st.stop()
if not health.get("model_loaded"):
    st.warning("The API is running, but its model is not loaded. Train the model first.")

try:
    sites = get_sites()
except requests.RequestException as error:
    st.error(f"Could not load sites from the API: {error}")
    st.stop()

if not sites:
    st.warning("No sites are available.")
    st.stop()

site_id = st.sidebar.selectbox("Select Site", sites)
try:
    site_info = get_site_info(site_id)
except requests.RequestException as error:
    st.error(f"Could not load information for {site_id}: {error}")
    st.stop()

feature_defaults = site_info["last_feature_row"]
last_date_in_data = pd.Timestamp(site_info["last_date_in_data"])
categorical_options = site_info.get("categorical_options", {})
available_dates = site_info.get("available_dates")

st.sidebar.markdown("---")
st.sidebar.markdown("**Site Info**")
st.sidebar.markdown(f"- Region: {site_info['region']}")
st.sidebar.markdown(f"- Cement Type: {site_info['cement_type']}")
st.sidebar.markdown(f"- Behavior: {site_info['behavior']}")
st.sidebar.markdown(f"- Silo Capacity: {site_info['silo_capacity']:.0f} tonnes")
st.sidebar.markdown(f"- Last Date in Data: {last_date_in_data.date()}")

st.sidebar.markdown("---")
forecast_mode = st.sidebar.radio(
    "Forecast Mode",
    ["From last date in data", "Custom start date (within data)", "Future dates (beyond data)"],
)
horizon = st.sidebar.slider("Forecast Horizon (weeks)", 1, 12, DEFAULT_HORIZON)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Edit Training Features")
st.sidebar.caption("Values override the feature row used by the API")

with st.sidebar.expander("📊 Lag & Rolling Features", expanded=False):
    lag_1 = st.number_input("consumed_tonnes_lag_1", value=float(feature_defaults["consumed_tonnes_lag_1"]), min_value=0.0, step=1.0)
    lag_2 = st.number_input("consumed_tonnes_lag_2", value=float(feature_defaults["consumed_tonnes_lag_2"]), min_value=0.0, step=1.0)
    lag_4 = st.number_input("consumed_tonnes_lag_4", value=float(feature_defaults["consumed_tonnes_lag_4"]), min_value=0.0, step=1.0)
    lag_8 = st.number_input("consumed_tonnes_lag_8", value=float(feature_defaults["consumed_tonnes_lag_8"]), min_value=0.0, step=1.0)
    roll_4 = st.number_input("consumed_tonnes_rollmean_4", value=float(feature_defaults["consumed_tonnes_rollmean_4"]), min_value=0.0, step=1.0)
    roll_8 = st.number_input("consumed_tonnes_rollmean_8", value=float(feature_defaults["consumed_tonnes_rollmean_8"]), min_value=0.0, step=1.0)

with st.sidebar.expander("🌦️ Exogenous Numeric", expanded=True):
    planned_pour = st.number_input("planned_pour_tonnes", value=float(feature_defaults["planned_pour_tonnes"]), min_value=0.0, step=10.0)
    rain_mm = st.number_input("rain_mm", value=float(feature_defaults["rain_mm"]), min_value=0.0, step=1.0)
    avg_temp = st.number_input("avg_temp_c", value=float(feature_defaults["avg_temp_c"]), step=0.5)
    silo_capacity = st.number_input("silo_capacity", value=float(feature_defaults["silo_capacity"]), min_value=0.0, step=10.0)

with st.sidebar.expander("🏷️ Categorical Features", expanded=False):
    # Uses dropdowns if the API supplies "categorical_options"; otherwise
    # falls back to the free-text inputs app1.py uses today.
    if categorical_options.get("behavior"):
        options = sorted(categorical_options["behavior"])
        default = feature_defaults["behavior"]
        behavior = st.selectbox("behavior", options, index=options.index(default) if default in options else 0)
    else:
        behavior = st.text_input("behavior", value=feature_defaults["behavior"])

    if categorical_options.get("cement_type"):
        options = sorted(categorical_options["cement_type"])
        default = feature_defaults["cement_type"]
        cement_type = st.selectbox("cement_type", options, index=options.index(default) if default in options else 0)
    else:
        cement_type = st.text_input("cement_type", value=feature_defaults["cement_type"])

    if categorical_options.get("region"):
        options = sorted(categorical_options["region"])
        default = feature_defaults["region"]
        region = st.selectbox("region", options, index=options.index(default) if default in options else 0)
    else:
        region = st.text_input("region", value=feature_defaults["region"])


def build_feature_overrides():
    return {
        "consumed_tonnes_lag_1": lag_1,
        "consumed_tonnes_lag_2": lag_2,
        "consumed_tonnes_lag_4": lag_4,
        "consumed_tonnes_lag_8": lag_8,
        "consumed_tonnes_rollmean_4": roll_4,
        "consumed_tonnes_rollmean_8": roll_8,
        "planned_pour_tonnes": planned_pour,
        "rain_mm": rain_mm,
        "avg_temp_c": avg_temp,
        "silo_capacity": silo_capacity,
        "behavior": behavior,
        "cement_type": cement_type,
        "region": region,
    }


def request_forecast(mode, start_date, scenario=None):
    payload = {
        "site_id": site_id,
        "start_date": pd.Timestamp(start_date).isoformat(),
        "horizon": horizon,
        "mode": mode,
        "feature_overrides": build_feature_overrides(),
    }
    if scenario is not None:
        payload["scenario"] = scenario
    return pd.DataFrame(api_request("POST", "/forecast", json=payload)["forecasts"])


scenario = None
if forecast_mode == "From last date in data":
    mode = "from_last_date"
    start_date = last_date_in_data
elif forecast_mode == "Custom start date (within data)":
    mode = "custom_date"
    # Uses a dropdown of real in-data dates if the API supplies
    # "available_dates"; otherwise falls back to app1.py's date picker.
    if available_dates:
        parsed_dates = sorted(pd.Timestamp(d) for d in available_dates)
        start_date = st.sidebar.selectbox(
            "Start Date", parsed_dates, index=len(parsed_dates) - 1,
            format_func=lambda d: d.date().isoformat(),
        )
    else:
        start_date = st.sidebar.date_input(
            "Start Date", value=last_date_in_data.date(), max_value=last_date_in_data.date()
        )
else:
    mode = "future_dates"
    start_date = st.sidebar.date_input(
        "Forecast Start Date",
        value=(last_date_in_data + timedelta(weeks=1)).date(),
        min_value=(last_date_in_data + timedelta(weeks=1)).date(),
    )
    future_weeks = pd.date_range(start=start_date, periods=horizon, freq="W")
    scenario_df = pd.DataFrame({
        "date": future_weeks,
        "planned_pour_tonnes": planned_pour,
        "rain_mm": rain_mm,
        "avg_temp_c": avg_temp,
    })
    st.markdown("**Per-week exogenous scenario**")
    edited = st.data_editor(scenario_df, use_container_width=True, hide_index=True, disabled=["date"])
    scenario = [
        {**row, "date": pd.Timestamp(row["date"]).isoformat()}
        for row in edited.to_dict("records")
    ]

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Forecast Results")
    if st.button("Generate Forecast", type="primary"):
        try:
            forecast_df = request_forecast(mode, start_date, scenario)
            forecast_df["date"] = pd.to_datetime(forecast_df["date"])
            forecast_df["forecast_consumed_tonnes"] = forecast_df["forecast_consumed_tonnes"].round(2)
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
            st.line_chart(forecast_df.set_index("date")["forecast_consumed_tonnes"], use_container_width=True)
            st.download_button(
                "📥 Download Forecast as CSV",
                forecast_df.to_csv(index=False),
                file_name=f"forecast_{site_id}_{pd.Timestamp.now():%Y%m%d}.csv",
                mime="text/csv",
            )
        except requests.HTTPError as error:
            detail = error.response.json().get("detail", str(error)) if error.response is not None else str(error)
            st.error(f"Forecast failed: {detail}")
        except requests.RequestException as error:
            st.error(f"Forecast failed: {error}")

with col2:
    # Historical Context panel: only shows up once the API exposes
    # GET /sites/{site_id}/history. Until then, this is silently skipped.
    history_df = get_site_history(site_id)
    if history_df is not None and not history_df.empty:
        st.subheader("📊 Historical Context")
        history_df["date"] = pd.to_datetime(history_df["date"])
        recent = history_df.sort_values("date").tail(8).copy()
        display_recent = recent.copy()
        display_recent["date"] = display_recent["date"].dt.date
        if "consumed_tonnes" in display_recent:
            display_recent["consumed_tonnes"] = display_recent["consumed_tonnes"].round(2)
        st.dataframe(display_recent, use_container_width=True, hide_index=True)
        if "consumed_tonnes" in recent:
            st.line_chart(recent.set_index("date")["consumed_tonnes"], use_container_width=True)
        st.caption("Last 8 weeks actual consumption")

st.markdown("---")
st.subheader("🔧 Feature Row Used for Forecasting")
st.dataframe(pd.DataFrame([{"site_id": site_id, **build_feature_overrides()}]), use_container_width=True, hide_index=True)
st.caption("Cement Demand Forecasting Model | Random Forest | API inference")
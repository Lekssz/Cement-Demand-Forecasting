"""Streamlit dashboard for the FastAPI inference service."""
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


health = get_health()
if health.get("status") == "down":
    st.error(f"Could not reach the forecasting API at {API_URL}.")
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
    behavior = st.text_input("behavior", value=feature_defaults["behavior"])
    cement_type = st.text_input("cement_type", value=feature_defaults["cement_type"])
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
    start_date = st.sidebar.date_input("Start Date", value=last_date_in_data.date(), max_value=last_date_in_data.date())
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

st.markdown("---")
st.subheader("🔧 Feature Row Used for Forecasting")
st.dataframe(pd.DataFrame([{"site_id": site_id, **build_feature_overrides()}]), use_container_width=True, hide_index=True)
st.caption("Cement Demand Forecasting Model | Random Forest | API inference")

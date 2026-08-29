"""
Plotly Dash business dashboard for cement demand forecasting
and inventory recommendations.

The dashboard calls the production ARIMAX FastAPI endpoints.
"""

import os
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
from dash import (
    Dash,
    Input,
    Output,
    State,
    callback,
    dash_table,
    dcc,
    html,
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

DEFAULT_HORIZON = 8
DEFAULT_DAILY_PLAN = 0.0


# ---------------------------------------------------------
# API HELPERS
# ---------------------------------------------------------
def api_request(
    method: str,
    path: str,
    **kwargs,
):
    response = requests.request(
        method,
        f"{API_URL}{path}",
        timeout=60,
        **kwargs,
    )

    if not response.ok:
        try:
            detail = response.json().get(
                "detail",
                response.text,
            )
        except ValueError:
            detail = response.text

        raise RuntimeError(
            f"API {response.status_code}: {detail}"
        )

    return response.json()


def load_sites():
    try:
        return api_request(
            "GET",
            "/arimax/sites",
        )
    except Exception:
        return []


# ---------------------------------------------------------
# DASH APP
# ---------------------------------------------------------
app = Dash(
    __name__,
    title="MIG Cement Demand Forecasting",
)

server = app.server


CARD_STYLE = {
    "backgroundColor": "white",
    "border": "1px solid #e5e7eb",
    "borderRadius": "12px",
    "padding": "18px",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
}

LABEL_STYLE = {
    "fontSize": "12px",
    "fontWeight": "600",
    "color": "#6b7280",
    "textTransform": "uppercase",
    "letterSpacing": "0.04em",
}

VALUE_STYLE = {
    "fontSize": "28px",
    "fontWeight": "700",
    "marginTop": "6px",
    "color": "#111827",
}


def metric_card(
    label,
    value_id,
    initial="—",
):
    return html.Div(
        [
            html.Div(
                label,
                style=LABEL_STYLE,
            ),
            html.Div(
                initial,
                id=value_id,
                style=VALUE_STYLE,
            ),
        ],
        style=CARD_STYLE,
    )


sites = load_sites()


app.layout = html.Div(
    [
        dcc.Store(
            id="site-config-store"
        ),

        # -------------------------------------------------
        # HEADER
        # -------------------------------------------------
        html.Div(
            [
                html.Div(
                    [
                        html.H1(
                            "MIG Cement Demand Forecasting",
                            style={
                                "margin": "0",
                                "fontSize": "30px",
                            },
                        ),
                        html.P(
                            "ARIMAX demand forecast and "
                            "3-day inventory reorder recommendations",
                            style={
                                "margin": "6px 0 0",
                                "color": "#6b7280",
                            },
                        ),
                    ]
                ),
                html.Div(
                    "Production model: ARIMAX(0,1,1)",
                    style={
                        "padding": "8px 12px",
                        "backgroundColor": "#eef2ff",
                        "borderRadius": "999px",
                        "fontWeight": "600",
                        "fontSize": "13px",
                    },
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "marginBottom": "24px",
            },
        ),
        # ---------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------
html.Div(
    [
        html.H2(
            "Model Evaluation",
            style={
                "marginTop": "0",
                "marginBottom": "6px",
            },
        ),
        html.P(
            (
                "Compare forecasting models and inspect "
                "ARIMAX backtest performance across sites."
            ),
            style={
                "color": "#6b7280",
                "marginTop": "0",
                "marginBottom": "24px",
            },
        ),

        # -------------------------------------------------
        # MODEL COMPARISON
        # -------------------------------------------------
        html.Div(
            [
                html.H3(
                    "Overall Model Comparison",
                    style={"marginTop": "0"},
                ),
                html.P(
                    (
                        "Mean MAPE across the five "
                        "8-week backtest periods."
                    ),
                    style={
                        "color": "#6b7280",
                    },
                ),
                dcc.Graph(
                    id="model-comparison-chart",
                    config={
                        "displayModeBar": False,
                    },
                ),
            ],
            style={
                **CARD_STYLE,
                "marginBottom": "20px",
            },
        ),

        # -------------------------------------------------
        # SITE PERFORMANCE
        # -------------------------------------------------
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Performance by Site",
                                    style={
                                        "margin": "0",
                                    },
                                ),
                                html.P(
                                    (
                                        "Compare site-level "
                                        "forecasting performance."
                                    ),
                                    style={
                                        "color": "#6b7280",
                                        "marginBottom": "0",
                                    },
                                ),
                            ]
                        ),

                        html.Div(
                            [
                                html.Label(
                                    "Model",
                                    style={
                                        "fontWeight": "600",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="evaluation-model-dropdown",
                                    options=[
                                        {
                                            "label": "ARIMAX",
                                            "value": "arimax",
                                        },
                                        {
                                            "label": "Random Forest",
                                            "value": "random-forest",
                                        },
                                    ],
                                    value="arimax",
                                    clearable=False,
                                    style={
                                        "width": "220px",
                                    },
                                ),
                            ]
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "marginBottom": "16px",
                    },
                ),

                html.Div(
                    id="site-performance-summary",
                    style={
                        "marginBottom": "12px",
                        "fontWeight": "600",
                    },
                ),

                dcc.Graph(
                    id="site-performance-chart",
                    config={
                        "displayModeBar": False,
                    },
                ),
            ],
            style={
                **CARD_STYLE,
                "marginBottom": "20px",
            },
        ),

        # -------------------------------------------------
        # ARIMAX BACKTEST ANALYSIS
        # -------------------------------------------------
        html.Div(
            [
                html.H3(
                    "ARIMAX Backtest Analysis",
                    style={"marginTop": "0"},
                ),
                html.P(
                    (
                        "Performance across the five "
                        "8-week historical backtest periods."
                    ),
                    style={
                        "color": "#6b7280",
                    },
                ),

                dcc.Graph(
                    id="arimax-window-chart",
                    config={
                        "displayModeBar": False,
                    },
                ),
            ],
            style={
                **CARD_STYLE,
                "marginBottom": "20px",
            },
        ),

        # -------------------------------------------------
        # ACTUAL VS FORECAST
        # -------------------------------------------------
        html.Div(
            [
                html.H3(
                    "Historical Actual vs Forecast",
                    style={"marginTop": "0"},
                ),

                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(
                                    "Site",
                                    style={
                                        "fontWeight": "600",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="backtest-site-dropdown",
                                    options=[
                                        {
                                            "label": site,
                                            "value": site,
                                        }
                                        for site in sites
                                    ],
                                    value=(
                                        sites[0]
                                        if sites
                                        else None
                                    ),
                                    clearable=False,
                                ),
                            ]
                        ),

                        html.Div(
                            [
                                html.Label(
                                    "Backtest Window",
                                    style={
                                        "fontWeight": "600",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="backtest-window-dropdown",
                                    options=[
                                        {
                                            "label": f"Window {i}",
                                            "value": i,
                                        }
                                        for i in range(1, 6)
                                    ],
                                    value=1,
                                    clearable=False,
                                ),
                            ]
                        ),
                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "20px",
                        "marginBottom": "16px",
                    },
                ),

                html.Div(
                    id="backtest-site-summary",
                    style={
                        "marginBottom": "10px",
                        "fontWeight": "600",
                    },
                ),

                dcc.Graph(
                    id="actual-vs-forecast-chart",
                    config={
                        "displayModeBar": False,
                    },
                ),
            ],
            style=CARD_STYLE,
        ),
    ],
    style={
        "marginTop": "32px",
    },
),
        # -------------------------------------------------
        # INPUT PANEL
        # -------------------------------------------------
        html.Div(
            [
                html.H3(
                    "Planning Inputs",
                    style={"marginTop": "0"},
                ),

                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(
                                    "Site",
                                    style={
                                        "fontWeight": "600"
                                    },
                                ),
                                dcc.Dropdown(
                                    id="site-dropdown",
                                    options=[
                                        {
                                            "label": s,
                                            "value": s,
                                        }
                                        for s in sites
                                    ],
                                    value=(
                                        sites[0]
                                        if sites
                                        else None
                                    ),
                                    clearable=False,
                                ),
                            ]
                        ),

                        html.Div(
                            [
                                html.Label(
                                    "Current physical inventory (t)",
                                    style={
                                        "fontWeight": "600"
                                    },
                                ),
                                dcc.Input(
                                    id="current-inventory",
                                    type="number",
                                    min=0,
                                    step=1,
                                    value=50,
                                    style={
                                        "width": "100%",
                                        "padding": "10px",
                                        "boxSizing": "border-box",
                                    },
                                ),
                            ]
                        ),

                        html.Div(
                            [
                                html.Label(
                                    "Forecast horizon (weeks)",
                                    style={
                                        "fontWeight": "600"
                                    },
                                ),
                                dcc.Slider(
                                    id="horizon-slider",
                                    min=1,
                                    max=8,
                                    step=1,
                                    value=DEFAULT_HORIZON,
                                    marks={
                                        i: str(i)
                                        for i in range(1, 9)
                                    },
                                ),
                            ]
                        ),
                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": (
                            "1fr 1fr 1.2fr"
                        ),
                        "gap": "20px",
                    },
                ),

                html.Div(
                    id="site-config-text",
                    style={
                        "marginTop": "16px",
                        "fontSize": "14px",
                        "color": "#4b5563",
                    },
                ),
            ],
            style={
                **CARD_STYLE,
                "marginBottom": "20px",
            },
        ),

        # -------------------------------------------------
        # DAILY POUR SCHEDULE
        # -------------------------------------------------
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Future Pour Schedule",
                                    style={
                                        "margin": "0"
                                    },
                                ),
                                html.P(
                                    "Enter the planned cement pour for each future day. "
                                    "Use 0 for days with no scheduled pour.",
                                    style={
                                        "color": "#6b7280",
                                        "marginBottom": "0",
                                    },
                                ),
                            ]
                        ),
                        html.Button(
                            "Generate Forecast",
                            id="generate-button",
                            n_clicks=0,
                            style={
                                "backgroundColor": "#111827",
                                "color": "white",
                                "border": "none",
                                "borderRadius": "8px",
                                "padding": "11px 18px",
                                "fontWeight": "600",
                                "cursor": "pointer",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "marginBottom": "16px",
                    },
                ),

                dash_table.DataTable(
                    id="daily-plan-table",
                    columns=[
                        {
                            "name": "Date",
                            "id": "date",
                            "editable": False,
                        },
                        {
                            "name": "Planned Pour (t)",
                            "id": "planned_pour_tonnes",
                            "type": "numeric",
                            "editable": True,
                        },
                    ],
                    data=[],
                    editable=True,
                    page_size=14,
                    style_table={
                        "overflowX": "auto",
                    },
                    style_header={
                        "fontWeight": "700",
                        "backgroundColor": "#f9fafb",
                    },
                    style_cell={
                        "padding": "9px",
                        "fontFamily": (
                            "-apple-system, BlinkMacSystemFont, "
                            "'Segoe UI', sans-serif"
                        ),
                        "textAlign": "left",
                    },
                ),
            ],
            style={
                **CARD_STYLE,
                "marginBottom": "20px",
            },
        ),

        # -------------------------------------------------
        # ERROR / STATUS
        # -------------------------------------------------
        html.Div(
            id="status-message",
            style={
                "marginBottom": "16px",
                "fontWeight": "600",
            },
        ),

        # -------------------------------------------------
        # KPI CARDS
        # -------------------------------------------------
        html.Div(
            [
                metric_card(
                    "Forecast Demand",
                    "forecast-total",
                ),
                metric_card(
                    "Inventory Utilisation",
                    "utilisation-value",
                ),
                metric_card(
                    "Reorder Status",
                    "reorder-status",
                ),
                metric_card(
                    "Recommended Order",
                    "recommended-order",
                ),
                metric_card(
                    "Target Inventory",
                    "target-inventory",
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": (
                    "repeat(5, minmax(0, 1fr))"
                ),
                "gap": "14px",
                "marginBottom": "20px",
            },
        ),

        # -------------------------------------------------
        # FORECAST CHART + INVENTORY DETAILS
        # -------------------------------------------------
        html.Div(
            [
                html.Div(
                    [
                        html.H3(
                            "8-Week Demand Forecast"
                        ),
                        dcc.Graph(
                            id="forecast-chart",
                            config={
                                "displayModeBar": False
                            },
                        ),
                    ],
                    style={
                        **CARD_STYLE,
                        "minWidth": "0",
                    },
                ),

                html.Div(
                    [
                        html.H3(
                            "Inventory Recommendation"
                        ),
                        html.Div(
                            id="inventory-details",
                            children=(
                                "Generate a forecast to "
                                "view the recommendation."
                            ),
                            style={
                                "lineHeight": "1.9",
                                "color": "#374151",
                            },
                        ),
                    ],
                    style=CARD_STYLE,
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": (
                    "2fr 1fr"
                ),
                "gap": "20px",
                "marginBottom": "20px",
            },
        ),

        # -------------------------------------------------
        # DAILY FORECAST TABLE
        # -------------------------------------------------
        html.Div(
            [
                html.H3(
                    "Daily Forecast Detail"
                ),
                dash_table.DataTable(
                    id="daily-forecast-table",
                    columns=[
                        {
                            "name": "Date",
                            "id": "date",
                        },
                        {
                            "name": "Planned Pour (t)",
                            "id": "planned_pour_tonnes",
                        },
                        {
                            "name": "Forecast Demand (t)",
                            "id": "forecast_consumed_tonnes",
                        },
                    ],
                    data=[],
                    page_size=14,
                    style_table={
                        "overflowX": "auto",
                    },
                    style_header={
                        "fontWeight": "700",
                        "backgroundColor": "#f9fafb",
                    },
                    style_cell={
                        "padding": "9px",
                        "textAlign": "left",
                    },
                ),
            ],
            style=CARD_STYLE,
        ),
    ],
    style={
        "maxWidth": "1500px",
        "margin": "0 auto",
        "padding": "28px",
        "fontFamily": (
            "-apple-system, BlinkMacSystemFont, "
            "'Segoe UI', sans-serif"
        ),
        "backgroundColor": "#f3f4f6",
        "minHeight": "100vh",
        "color": "#111827",
    },
)


# ---------------------------------------------------------
# CALLBACK: LOAD SITE CONFIG
# ---------------------------------------------------------
@callback(
    Output(
        "site-config-store",
        "data",
    ),
    Output(
        "site-config-text",
        "children",
    ),
    Output(
        "current-inventory",
        "max",
    ),
    Input(
        "site-dropdown",
        "value",
    ),
)
def load_site_config(site_id):
    if not site_id:
        return {}, "No site selected.", None

    try:
        config = api_request(
            "GET",
            f"/arimax/sites/{site_id}/config",
        )

        text = (
            f"Silo capacity: "
            f"{config['silo_capacity']:.0f} t  |  "
            f"Safety stock: "
            f"{config['safety_stock_tonnes']:.2f} t  |  "
            f"Next forecast week: "
            f"{pd.Timestamp(config['next_forecast_week']).date()}"
        )

        return (
            config,
            text,
            config["silo_capacity"],
        )

    except Exception as exc:
        return (
            {},
            f"Could not load site configuration: {exc}",
            None,
        )


# ---------------------------------------------------------
# CALLBACK: BUILD EDITABLE DAILY SCHEDULE
# ---------------------------------------------------------
@callback(
    Output(
        "daily-plan-table",
        "data",
    ),
    Input(
        "site-config-store",
        "data",
    ),
    Input(
        "horizon-slider",
        "value",
    ),
)
def build_daily_plan(
    config,
    horizon,
):
    if not config or not horizon:
        return []

    start = pd.Timestamp(
        config["next_forecast_week"]
    ).normalize()

    dates = pd.date_range(
        start=start,
        periods=int(horizon) * 7,
        freq="D",
    )

    return [
        {
            "date": date.date().isoformat(),
            "planned_pour_tonnes": (
                DEFAULT_DAILY_PLAN
            ),
        }
        for date in dates
    ]


# ---------------------------------------------------------
# CALLBACK: GENERATE FORECAST
# ---------------------------------------------------------
@callback(
    Output(
        "status-message",
        "children",
    ),
    Output(
        "status-message",
        "style",
    ),
    Output(
        "forecast-total",
        "children",
    ),
    Output(
        "utilisation-value",
        "children",
    ),
    Output(
        "reorder-status",
        "children",
    ),
    Output(
        "recommended-order",
        "children",
    ),
    Output(
        "target-inventory",
        "children",
    ),
    Output(
        "forecast-chart",
        "figure",
    ),
    Output(
        "inventory-details",
        "children",
    ),
    Output(
        "daily-forecast-table",
        "data",
    ),
    Input(
        "generate-button",
        "n_clicks",
    ),
    State(
        "site-dropdown",
        "value",
    ),
    State(
        "current-inventory",
        "value",
    ),
    State(
        "daily-plan-table",
        "data",
    ),
    prevent_initial_call=True,
)
def generate_forecast(
    n_clicks,
    site_id,
    current_inventory,
    daily_plan,
):
    empty_fig = go.Figure()

    if (
        not site_id
        or current_inventory is None
        or not daily_plan
    ):
        return (
            "Complete the planning inputs first.",
            {
                "color": "#b91c1c",
                "marginBottom": "16px",
                "fontWeight": "600",
            },
            "—",
            "—",
            "—",
            "—",
            "—",
            empty_fig,
            "No recommendation available.",
            [],
        )

    try:
        payload = {
            "site_id": site_id,
            "current_inventory_tonnes": (
                float(current_inventory)
            ),
            "daily_plan": [
                {
                    "date": (
                        pd.Timestamp(
                            row["date"]
                        ).isoformat()
                    ),
                    "planned_pour_tonnes": (
                        float(
                            row[
                                "planned_pour_tonnes"
                            ]
                            or 0
                        )
                    ),
                }
                for row in daily_plan
            ],
        }

        result = api_request(
            "POST",
            "/arimax/forecast-inventory",
            json=payload,
        )

        weekly = pd.DataFrame(
            result["weekly_forecast"]
        )

        daily = pd.DataFrame(
            result["daily_forecast"]
        )

        inventory = result[
            "inventory_recommendation"
        ]

        weekly["week_start"] = (
            pd.to_datetime(
                weekly["week_start"]
            )
        )

        total_forecast = float(
            weekly[
                "forecast_consumed_tonnes"
            ].sum()
        )

        figure = go.Figure()

        figure.add_trace(
            go.Scatter(
                x=weekly["week_start"],
                y=weekly[
                    "forecast_consumed_tonnes"
                ],
                mode="lines+markers",
                name="Forecast demand",
            )
        )

        figure.add_trace(
            go.Scatter(
                x=weekly["week_start"],
                y=weekly[
                    "planned_pour_tonnes"
                ],
                mode="lines+markers",
                name="Planned pour",
                line={
                    "dash": "dash"
                },
            )
        )

        figure.update_layout(
            margin={
                "l": 20,
                "r": 20,
                "t": 20,
                "b": 20,
            },
            xaxis_title="Week",
            yaxis_title="Tonnes",
            hovermode="x unified",
            legend={
                "orientation": "h",
                "y": 1.1,
            },
        )

        reorder = bool(
            inventory["reorder_alert"]
        )

        reorder_text = (
            "REORDER"
            if reorder
            else "NO REORDER"
        )

        details = html.Div(
            [
                html.Div(
                    [
                        html.Strong(
                            "Reorder trigger: "
                        ),
                        f"{inventory['reorder_trigger_tonnes']:.2f} t",
                    ]
                ),
                html.Div(
                    [
                        html.Strong(
                            "Safety stock: "
                        ),
                        f"{inventory['safety_stock_tonnes']:.2f} t",
                    ]
                ),
                html.Div(
                    [
                        html.Strong(
                            "3-day demand cover: "
                        ),
                        f"{inventory['coverage_forecast_tonnes']:.2f} t",
                    ]
                ),
                html.Div(
                    [
                        html.Strong(
                            "Projected stock after order: "
                        ),
                        f"{inventory['projected_inventory_after_order_tonnes']:.2f} t",
                    ]
                ),
                html.Div(
                    [
                        html.Strong(
                            "Remaining silo capacity: "
                        ),
                        f"{inventory['remaining_capacity_after_order_tonnes']:.2f} t",
                    ]
                ),
            ]
        )

        daily_display = daily[
            [
                "date",
                "planned_pour_tonnes",
                "forecast_consumed_tonnes",
            ]
        ].copy()

        daily_display["date"] = (
            pd.to_datetime(
                daily_display["date"]
            )
            .dt.date
            .astype(str)
        )

        daily_display[
            "planned_pour_tonnes"
        ] = (
            daily_display[
                "planned_pour_tonnes"
            ]
            .round(2)
        )

        daily_display[
            "forecast_consumed_tonnes"
        ] = (
            daily_display[
                "forecast_consumed_tonnes"
            ]
            .round(2)
        )

        return (
            "Forecast generated successfully.",
            {
                "color": "#047857",
                "marginBottom": "16px",
                "fontWeight": "600",
            },
            f"{total_forecast:,.1f} t",
            (
                f"{inventory['inventory_utilisation_pct']:.1f}%"
            ),
            reorder_text,
            (
                f"{inventory['recommended_order_tonnes']:.1f} t"
            ),
            (
                f"{inventory['target_inventory_tonnes']:.1f} t"
            ),
            figure,
            details,
            daily_display.to_dict(
                "records"
            ),
        )

    except Exception as exc:
        return (
            f"Forecast failed: {exc}",
            {
                "color": "#b91c1c",
                "marginBottom": "16px",
                "fontWeight": "600",
            },
            "—",
            "—",
            "—",
            "—",
            "—",
            empty_fig,
            "No recommendation available.",
            [],
        )
# ---------------------------------------------------------
# CALLBACK: OVERALL MODEL COMPARISON
# ---------------------------------------------------------
# Load the mean backtest MAPE for ARIMAX and Random Forest
# from the model-evaluation API.
#
# This chart explains why ARIMAX was selected as the
# production forecasting model.
@callback(
    Output(
        "model-comparison-chart",
        "figure",
    ),
    Input(
        "model-comparison-chart",
        "id",
    ),
)
def update_model_comparison(_):

    try:
        result = api_request(
            "GET",
            "/model-evaluation/summary",
        )

        models = result["models"]

        model_names = [
            item["model"]
            for item in models
        ]

        mape_values = [
            item["mean_mape"]
            for item in models
        ]

        roles = [
            item["role"]
            for item in models
        ]

        figure = go.Figure()

        figure.add_trace(
            go.Bar(
                x=model_names,
                y=mape_values,
                text=[
                    f"{value:.2f}%"
                    for value in mape_values
                ],
                textposition="outside",
                customdata=roles,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Mean MAPE: %{y:.2f}%<br>"
                    "Role: %{customdata}"
                    "<extra></extra>"
                ),
            )
        )

        # Project forecasting target = MAPE <= 15%.
        figure.add_hline(
            y=result["target_pct"],
            line_dash="dash",
            annotation_text="15% MAPE target",
        )

        figure.update_layout(
            title="Mean MAPE Across Five Backtest Windows",
            xaxis_title="Model",
            yaxis_title="MAPE (%)",
            showlegend=False,
            margin={
                "l": 50,
                "r": 30,
                "t": 70,
                "b": 50,
            },
        )

        return figure

    except Exception as exc:

        figure = go.Figure()

        figure.add_annotation(
            text=(
                "Could not load model comparison: "
                f"{exc}"
            ),
            showarrow=False,
        )

        return figure

# ---------------------------------------------------------
# CALLBACK: SITE-LEVEL MODEL PERFORMANCE
# ---------------------------------------------------------
# The dropdown switches between ARIMAX and Random Forest.
# The API returns MAPE for all 30 sites for the selected model.
@callback(
    Output(
        "site-performance-summary",
        "children",
    ),
    Output(
        "site-performance-chart",
        "figure",
    ),
    Input(
        "evaluation-model-dropdown",
        "value",
    ),
)
def update_site_performance(
    model_name,
):

    try:
        result = api_request(
            "GET",
            f"/model-evaluation/sites/{model_name}",
        )

        sites_data = result["sites"]

        site_ids = [
            item["site_id"]
            for item in sites_data
        ]

        mape_values = [
            item["mape"]
            for item in sites_data
        ]

        summary = (
            f"{result['sites_meeting_target']} of "
            f"{result['total_sites']} sites meet the "
            f"{result['target_pct']:.0f}% MAPE target."
        )

        figure = go.Figure()

        figure.add_trace(
            go.Bar(
                x=site_ids,
                y=mape_values,
                text=[
                    f"{value:.1f}%"
                    for value in mape_values
                ],
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "MAPE: %{y:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

        figure.add_hline(
            y=result["target_pct"],
            line_dash="dash",
            annotation_text="15% MAPE target",
        )

        figure.update_layout(
            title=(
                f"{result['model']} Forecast "
                "Performance by Site"
            ),
            xaxis_title="Site",
            yaxis_title="MAPE (%)",
            showlegend=False,
            margin={
                "l": 50,
                "r": 30,
                "t": 70,
                "b": 90,
            },
        )

        figure.update_xaxes(
            tickangle=-45,
        )

        return (
            summary,
            figure,
        )

    except Exception as exc:

        figure = go.Figure()

        figure.add_annotation(
            text=(
                "Could not load site performance: "
                f"{exc}"
            ),
            showarrow=False,
        )

        return (
            "Site performance unavailable.",
            figure,
        )
# ---------------------------------------------------------
# CALLBACK: ARIMAX BACKTEST WINDOWS
# ---------------------------------------------------------
# Show how ARIMAX performed across each of the five
# historical 8-week backtest periods.
@callback(
    Output(
        "arimax-window-chart",
        "figure",
    ),
    Input(
        "arimax-window-chart",
        "id",
    ),
)
def update_arimax_windows(_):

    try:
        result = api_request(
            "GET",
            "/model-evaluation/arimax/windows",
        )

        windows = result["windows"]

        window_labels = [
            f"W{item['window']}"
            for item in windows
        ]

        mape_values = [
            item["mape"]
            for item in windows
        ]

        figure = go.Figure()

        figure.add_trace(
            go.Bar(
                x=window_labels,
                y=mape_values,
                text=[
                    f"{value:.2f}%"
                    for value in mape_values
                ],
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "MAPE: %{y:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

        figure.add_hline(
            y=result["target_pct"],
            line_dash="dash",
            annotation_text="15% MAPE target",
        )

        figure.update_layout(
            title="ARIMAX MAPE Across Five Backtest Windows",
            xaxis_title="Backtest Window",
            yaxis_title="MAPE (%)",
            showlegend=False,
            margin={
                "l": 50,
                "r": 30,
                "t": 70,
                "b": 50,
            },
        )

        return figure

    except Exception as exc:

        figure = go.Figure()

        figure.add_annotation(
            text=(
                "Could not load ARIMAX backtests: "
                f"{exc}"
            ),
            showarrow=False,
        )

        return figure

# ---------------------------------------------------------
# CALLBACK: ACTUAL VS FORECAST BACKTEST
# ---------------------------------------------------------
# Show the historical demand and ARIMAX prediction for a
# selected site during one of the five 8-week backtests.
@callback(
    Output(
        "backtest-site-summary",
        "children",
    ),
    Output(
        "actual-vs-forecast-chart",
        "figure",
    ),
    Input(
        "backtest-site-dropdown",
        "value",
    ),
    Input(
        "backtest-window-dropdown",
        "value",
    ),
)
def update_backtest_detail(
    site_id,
    window,
):

    if not site_id or not window:
        return (
            "Select a site and backtest window.",
            go.Figure(),
        )

    try:
        # Get the eight weekly actual and forecast values.
        prediction_result = api_request(
            "GET",
            (
                f"/model-evaluation/arimax/site/"
                f"{site_id}/window/{window}"
            ),
        )

        # Get the site's MAPE for this particular window.
        performance_result = api_request(
            "GET",
            (
                f"/model-evaluation/arimax/site/"
                f"{site_id}/windows"
            ),
        )

        selected_window = next(
            item
            for item in performance_result["windows"]
            if item["window"] == int(window)
        )

        points = prediction_result["points"]

        dates = [
            item["week_start"]
            for item in points
        ]

        actual = [
            item["actual"]
            for item in points
        ]

        forecast = [
            item["forecast"]
            for item in points
        ]

        summary = (
            f"{site_id} | Window {window} | "
            f"MAPE: {selected_window['mape']:.2f}% | "
            f"MAE: {selected_window['mae']:.2f} t | "
            f"RMSE: {selected_window['rmse']:.2f} t"
        )

        figure = go.Figure()

        figure.add_trace(
            go.Scatter(
                x=dates,
                y=actual,
                mode="lines+markers",
                name="Actual Demand",
            )
        )

        figure.add_trace(
            go.Scatter(
                x=dates,
                y=forecast,
                mode="lines+markers",
                name="ARIMAX Forecast",
            )
        )

        figure.update_layout(
            title=(
                f"{site_id} — Window {window}: "
                "Actual vs Forecast"
            ),
            xaxis_title="Week",
            yaxis_title="Cement Demand (tonnes)",
            hovermode="x unified",
            margin={
                "l": 50,
                "r": 30,
                "t": 70,
                "b": 50,
            },
        )

        return (
            summary,
            figure,
        )

    except Exception as exc:

        figure = go.Figure()

        figure.add_annotation(
            text=(
                "Could not load backtest detail: "
                f"{exc}"
            ),
            showarrow=False,
        )

        return (
            "Backtest detail unavailable.",
            figure,
        )
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(
            os.getenv(
                "DASH_PORT",
                "8050",
            )
        ),
        debug=False,
    )

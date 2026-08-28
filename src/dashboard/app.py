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
    title="Cement Demand Forecasting",
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
                            "Cement Demand Forecasting",
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

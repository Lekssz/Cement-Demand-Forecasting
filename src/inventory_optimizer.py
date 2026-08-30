"""
Production inventory recommendation logic for the cement forecasting project.

The forecasting model supplies expected demand.
This module converts that demand forecast into:
- a reorder trigger,
- a reorder alert,
- a target inventory level,
- and a recommended order quantity.

Validated policy:
- trigger on today's expected demand + site safety stock
- if triggered, order up to 3 days of expected demand + safety stock
- never recommend stock above the stated silo capacity

Important:
Supplier lead time, truck capacity and minimum order quantity are not included
because those values are not available in the project data.
"""

from typing import Dict, Sequence

import numpy as np


DEFAULT_COVERAGE_DAYS = 3


# ---------------------------------------------------------
# 1. INPUT VALIDATION
# ---------------------------------------------------------
def _validate_inputs(
    current_inventory_tonnes: float,
    silo_capacity: float,
    daily_forecast_tonnes: Sequence[float],
    safety_stock_tonnes: float,
    coverage_days: int,
) -> None:
    """
    Validate the physical inventory inputs before calculating a recommendation.

    Production code should not silently accept impossible inventory values.
    """

    if silo_capacity <= 0:
        raise ValueError(
            "silo_capacity must be greater than 0."
        )

    if current_inventory_tonnes < 0:
        raise ValueError(
            "current_inventory_tonnes cannot be negative."
        )

    if current_inventory_tonnes > silo_capacity:
        raise ValueError(
            "current_inventory_tonnes cannot exceed silo_capacity. "
            "Pass a physical/capacity-constrained inventory value."
        )

    if safety_stock_tonnes < 0:
        raise ValueError(
            "safety_stock_tonnes cannot be negative."
        )

    if coverage_days < 1:
        raise ValueError(
            "coverage_days must be at least 1."
        )

    if len(daily_forecast_tonnes) == 0:
        raise ValueError(
            "daily_forecast_tonnes must contain at least one forecast."
        )


# ---------------------------------------------------------
# 2. CALCULATE INVENTORY RECOMMENDATION
# ---------------------------------------------------------
def calculate_inventory_recommendation(
    current_inventory_tonnes: float,
    silo_capacity: float,
    daily_forecast_tonnes: Sequence[float],
    safety_stock_tonnes: float,
    coverage_days: int = DEFAULT_COVERAGE_DAYS,
) -> Dict[str, float]:
    """
    Calculate the production reorder recommendation for one site.

    Parameters
    ----------
    current_inventory_tonnes:
        Current usable physical stock at the site.

    silo_capacity:
        Supplied usable silo-capacity value.

    daily_forecast_tonnes:
        Forecast demand beginning with today.
        The first `coverage_days` values are used for the target stock.

    safety_stock_tonnes:
        Site-specific safety stock calibrated from historical
        ARIMAX under-forecast errors.

    coverage_days:
        Number of forecast days to cover when a reorder is triggered.
        The validated project policy uses 3 days.

    Returns
    -------
    dict
        Values required by the dashboard/API.
    """

    _validate_inputs(
        current_inventory_tonnes=current_inventory_tonnes,
        silo_capacity=silo_capacity,
        daily_forecast_tonnes=daily_forecast_tonnes,
        safety_stock_tonnes=safety_stock_tonnes,
        coverage_days=coverage_days,
    )

    # Convert input to clean numeric values and prevent negative
    # forecasts from creating negative demand requirements.
    forecasts = np.maximum(
        np.asarray(
            daily_forecast_tonnes,
            dtype=float,
        ),
        0.0,
    )

    current_inventory = float(
        current_inventory_tonnes
    )

    capacity = float(
        silo_capacity
    )

    safety_stock = float(
        safety_stock_tonnes
    )


    # -----------------------------------------------------
    # 3. REORDER TRIGGER
    # -----------------------------------------------------
    # The trigger uses TODAY'S expected demand plus safety stock.
    today_expected_demand = float(
        forecasts[0]
    )

    reorder_trigger = min(
        capacity,
        today_expected_demand
        + safety_stock,
    )

    reorder_alert = (
        current_inventory
        <= reorder_trigger
    )


    # -----------------------------------------------------
    # 4. THREE-DAY TARGET INVENTORY
    # -----------------------------------------------------
    # If an alert fires, stock up to the next three days'
    # expected demand plus safety stock.
    available_horizon = min(
        coverage_days,
        len(forecasts),
    )

    coverage_demand = float(
        forecasts[
            :available_horizon
        ].sum()
    )

    unconstrained_target = (
        coverage_demand
        + safety_stock
    )

    target_inventory = min(
        capacity,
        unconstrained_target,
    )


    # -----------------------------------------------------
    # 5. RECOMMENDED ORDER QUANTITY
    # -----------------------------------------------------
    if reorder_alert:

        recommended_order = max(
            target_inventory
            - current_inventory,
            0.0,
        )

    else:

        recommended_order = 0.0

    # Final defensive capacity check.
    available_capacity = max(
        capacity
        - current_inventory,
        0.0,
    )

    recommended_order = min(
        recommended_order,
        available_capacity,
    )


    # -----------------------------------------------------
    # 6. DASHBOARD-FRIENDLY OUTPUT
    # -----------------------------------------------------
    utilisation_pct = (
        current_inventory
        / capacity
        * 100
    )

    projected_inventory_after_order = (
        current_inventory
        + recommended_order
    )

    return {
        "current_inventory_tonnes": round(
            current_inventory,
            2,
        ),
        "silo_capacity": round(
            capacity,
            2,
        ),
        "inventory_utilisation_pct": round(
            utilisation_pct,
            2,
        ),
        "today_expected_demand_tonnes": round(
            today_expected_demand,
            2,
        ),
        "coverage_days": int(
            coverage_days
        ),
        "coverage_forecast_tonnes": round(
            coverage_demand,
            2,
        ),
        "safety_stock_tonnes": round(
            safety_stock,
            2,
        ),
        "reorder_trigger_tonnes": round(
            reorder_trigger,
            2,
        ),
        "target_inventory_tonnes": round(
            target_inventory,
            2,
        ),
        "reorder_alert": bool(
            reorder_alert
        ),
        "recommended_order_tonnes": round(
            recommended_order,
            2,
        ),
        "projected_inventory_after_order_tonnes": round(
            projected_inventory_after_order,
            2,
        ),
        "remaining_capacity_after_order_tonnes": round(
            capacity
            - projected_inventory_after_order,
            2,
        ),
    }

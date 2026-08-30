# MIG Cement Demand Forecasting and Inventory Optimisation

## Introduction

This project develops a forecasting and inventory planning system for cement demand across multiple construction sites.

The objective is to use historical cement consumption, planned pour schedules, and other available operational information to forecast demand up to 8 weeks ahead. These forecasts are then combined with current inventory levels and silo capacity to support proactive reorder decisions.

The overall workflow is:

```text
Historical Consumption
+ Planned Pours
+ Operational Information
        ↓
8-Week Cement Demand Forecast
        ↓
Compare Forecast With Current Inventory
        ↓
Determine When and How Much to Reorder
        ↓
Support Pour Readiness and Stock Planning
```

## Project Targets

The project aims to achieve:

- Forecast MAPE ≤ 15%
- Pour readiness ≥ 98%
- 20% improvement in silo utilisation efficiency
- 30% reduction in material write-offs

The silo-utilisation and material write-off objectives remain business targets and are not claimed as achieved without an agreed operational baseline.

## Forecasting Approach

Several forecasting approaches were investigated, including Random Forest and ARIMAX.

The final production forecasting model is:

**ARIMAX(0,1,1)**

Planned cement pours are used as an external predictor because they provide useful information about expected future construction activity.

Five rolling 8-week backtest periods were used to evaluate forecasting performance.

| Model | Mean MAPE | Sites Meeting ≤15% MAPE | Role |
|---|---:|---:|---|
| ARIMAX | 11.36% | 22 / 30 | Production |
| Random Forest | 17.98% | 14 / 30 | Benchmark |

ARIMAX achieved a mean backtest MAPE below the project's 15% target and outperformed Random Forest across the evaluation periods. Random Forest is retained as a benchmark model.

Weather was also investigated as a potential predictor, particularly rainfall. However, the dataset does not contain sufficiently detailed site-level location information to obtain reliable future weather forecasts for each site.

Weather was therefore excluded from the deployable model rather than introducing arbitrary geographic assumptions.

## Site-Level Forecasting Insights

Forecasting difficulty is not evenly distributed across all sites.

Poorer forecasting performance is concentrated mainly among sites classified as **aggressive** or **chaotic**, while more stable sites are generally easier to forecast.

This shows why overall model performance should be considered alongside site-level performance. Although ARIMAX meets the overall forecasting target, some individual sites remain more difficult to predict.

## Inventory Data Investigation

During the inventory analysis, an important data-quality issue was identified.

Some conservative sites consume substantially less cement than other sites, while their recorded deliveries remain relatively high. This causes calculated historical inventory to accumulate and, in some cases, exceed the supplied silo capacity.

The underlying inventory accounting relationship is internally consistent, but the historical inventory records cannot always be interpreted as physically constrained silo stock.

For this reason, the original records were preserved rather than modified.

The production inventory optimiser instead uses:

- current physical inventory
- supplied silo capacity
- forecast demand
- site-specific safety stock

Silo capacity is taken directly from the supplied project data and is not predicted by the forecasting model.

## Inventory Optimisation

Forecast demand is combined with a safety-stock buffer to determine when inventory should be replenished.

The final policy uses a **3-day order-up-to approach**:

- A reorder alert is triggered when inventory falls below expected near-term demand plus safety stock.
- The recommended order raises inventory toward three days of forecast demand plus safety stock.
- The recommendation is capped by the site's silo capacity.

The policy was validated across multiple historical periods.

| Inventory Metric | Result |
|---|---:|
| Mean pour readiness | 99.92% |
| Minimum period readiness | 99.67% |
| Project readiness target | ≥ 98% |

The selected policy therefore exceeded the project's pour-readiness target during historical validation.

Supplier lead times, minimum order quantities and truck capacities were not available in the supplied data, so the recommendation should be interpreted as an inventory planning recommendation rather than a complete supplier delivery schedule.

## Application

The final application combines:

- ARIMAX cement-demand forecasting
- Random Forest benchmarking
- site-specific safety stock
- inventory reorder recommendations
- model evaluation and historical backtesting
- FastAPI services
- Plotly Dash dashboard
- MLflow experiment tracking
- Docker Compose deployment

The dashboard allows users to:

- select a construction site
- provide current physical inventory
- enter future planned pours
- generate forecasts up to 8 weeks ahead
- view inventory utilisation
- receive reorder alerts
- view recommended order quantities
- compare ARIMAX and Random Forest performance
- inspect site-level forecasting performance
- compare historical actual demand against ARIMAX forecasts

Detailed deployment instructions are available in:

```text
src/ReadMe.md
```
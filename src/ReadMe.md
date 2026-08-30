# MIG Cement Demand Forecasting — Deployment System

This directory contains the deployment stack for the MIG Cement Demand Forecasting application.

The system combines ARIMAX demand forecasting, Random Forest benchmarking, inventory recommendations, FastAPI, Plotly Dash, MLflow, and Docker Compose.

## Architecture

```text
Historical Data + Planned Pours
            ↓
      Forecast Models
      ├── ARIMAX
      └── Random Forest
            ↓
          FastAPI
       ↙           ↘
 Forecasting     Model Evaluation
      ↓
Inventory Recommendation
      ↓
   Plotly Dash

Random Forest Training
         ↓
       MLflow
```

ARIMAX(0,1,1) is the selected production forecasting model.

Random Forest is retained as a benchmark model.

## Services

| Service | Port | Description |
|---------|------|-------------|
| **mlflow** | 5001 locally | Experiment tracking |
| **model** | — | Random Forest training job |
| **api** | 8000 | Forecasting, inventory and evaluation API |
| **dash** | 8050 | Main Plotly Dash dashboard |
| **streamlit** | 8501 | Original Streamlit dashboard |

## Quick Start (Docker Compose)

### Prerequisites

- Docker and Docker Compose
- Processed data at `../data/processed/operations_cleaned.csv`
- Model evaluation reports under `../reports/`

### 1. Start MLflow tracking server

```bash
cd src
docker compose up -d mlflow
```

MLflow is available locally at:

```text
http://localhost:5001
```

### 2. Train the model

```bash
docker compose run --rm model
```

This runs the existing Random Forest training pipeline and saves the model artifact under `../models/`.

### 3. Start API + Dashboard

```bash
docker compose up -d api dash
```

Check container status:

```bash
docker compose ps
```

### 4. Access services

- **Plotly Dash:** `http://localhost:8050`
- **FastAPI Docs:** `http://localhost:8000/docs`
- **MLflow:** `http://localhost:5001`
- **Streamlit:** `http://localhost:8501` if running

### 5. Test the API

```bash
# Main API
curl http://localhost:8000/health

# ARIMAX production service
curl http://localhost:8000/arimax/health

# Model comparison
curl http://localhost:8000/model-evaluation/summary
```

## Local Development (without Docker)

### Install dependencies

From the project root:

```bash
./cement-env/bin/python -m pip install -r src/requirements.txt
```

### Set environment variables

```bash
export PROJECT_ROOT="$PWD"
export MLFLOW_TRACKING_URI=http://localhost:5001
export MLFLOW_EXPERIMENT=cement_demand_forecasting
```

### Start MLflow locally

Using Docker:

```bash
cd src
docker compose up -d mlflow
```

### Train model

From the project root:

```bash
./cement-env/bin/python -m src.pipeline \
  --cutoff-date 2022-12-31 \
  --output models/cement_demand_rf.pkl \
  --run-name "local-training"
```

### Run API

```bash
export PROJECT_ROOT="$PWD"

./cement-env/bin/python -m uvicorn \
  src.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload
```

### Run Dashboard

```bash
./cement-env/bin/python src/dashboard/app.py
```

Dashboard:

```text
http://localhost:8050
```

## Project Structure

```text
src/
├── api/
│   ├── main.py
│   ├── arimax_routes.py
│   └── model_evaluation_routes.py
│
├── dashboard/
│   └── app.py
│
├── streamlit/
│   └── app.py
│
├── common/
│   ├── config.py
│   ├── logging_config.py
│   └── schemas.py
│
├── arimax_forecaster.py
├── forecast_inventory_service.py
├── inventory_optimizer.py
├── inference.py
├── pipeline.py
├── preprocessing.py
├── data_loader.py
│
├── Dockerfile.api
├── Dockerfile.model
├── Dockerfile.dash
├── Dockerfile.streamlit
├── docker-compose.yml
└── requirements.txt
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_ROOT` | `/app` | Project root inside Docker |
| `DATA_PROCESSED` | `/app/data/processed` | Processed data |
| `MODEL_DIR` | `/app/models` | Model artifacts |
| `MLFLOW_TRACKING_URI` | — | MLflow server |
| `MLFLOW_EXPERIMENT` | `cement_demand_forecasting` | MLflow experiment |
| `API_PORT` | `8000` | FastAPI port |
| `DASH_PORT` | `8050` | Dash port |
| `API_URL` | `http://api:8000` | API used by Dash |
| `HORIZON` | `8` | Maximum forecast horizon |
| `LOG_LEVEL` | `INFO` | Logging level |

## API Endpoints

### Random Forest

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | API / RF health |
| `GET` | `/sites` | Site list |
| `POST` | `/forecast` | Random Forest forecast |

### ARIMAX Production

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/arimax/health` | ARIMAX health |
| `GET` | `/arimax/sites` | Available sites |
| `GET` | `/arimax/sites/{site_id}/config` | Site configuration |
| `POST` | `/arimax/forecast-inventory` | Forecast + inventory recommendation |

### Model Evaluation

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/model-evaluation/summary` | ARIMAX vs Random Forest |
| `GET` | `/model-evaluation/sites/{model_name}` | Site-level performance |
| `GET` | `/model-evaluation/arimax/windows` | ARIMAX backtests |
| `GET` | `/model-evaluation/arimax/site/{site_id}/window/{window}` | Actual vs forecast |

## MLflow Integration

MLflow is used by the Random Forest training pipeline to record:

- model parameters
- MAE and MAPE
- training configuration
- model artifacts

ARIMAX is deployed separately through the forecasting service.

## CI/CD Notes

Each service has its own Dockerfile:

- `Dockerfile.model` — model training
- `Dockerfile.api` — FastAPI
- `Dockerfile.dash` — Plotly Dash
- `Dockerfile.streamlit` — original Streamlit dashboard

Docker Compose is used to build and run the integrated application.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API unavailable | `docker compose logs api` |
| Dash unavailable | `docker compose logs dash` |
| Model evaluation unavailable | Check the `reports` volume |
| Dataset unavailable | Check `../data/processed/operations_cleaned.csv` |
| MLflow unavailable | `docker compose logs mlflow` |
| Port 5000 conflict on macOS | Use local MLflow port `5001` |

## Data Requirements

The application expects:

```text
../data/processed/operations_cleaned.csv
```

Important fields include:

- `date`
- `site_id`
- `cement_type`
- `planned_pour_tonnes`
- `consumed_tonnes`
- `opening_inventory_tonnes`
- `deliveries_tonnes`
- `closing_inventory_tonnes`
- `rain_mm`
- `avg_temp_c`
- `silo_capacity`
- `behavior`
- `region`

The production ARIMAX service aggregates the daily data into weekly site-level demand and forecasts up to 8 weeks ahead.
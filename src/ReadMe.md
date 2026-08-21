# Cement Demand Forecasting — Deployment System

This directory contains the production-ready deployment stack for the cement demand forecasting model.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Model     │────▶│    API      │◀───▶│  Streamlit  │
│  Training   │     │  (FastAPI)  │     │  Dashboard  │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
   MLflow              Model Artifact      User Interface
   Tracking            (joblib .pkl)       (port 8501)
   (port 5000)         (port 8000)
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| **mlflow** | 5000 | Experiment tracking UI & model registry |
| **model** | — | Training job (runs once, exits) |
| **api** | 8000 | FastAPI inference service (`/forecast`, `/health`, `/sites`) |
| **streamlit** | 8501 | Interactive dashboard calling the API |

## Quick Start (Docker Compose)

### Prerequisites
- Docker & Docker Compose installed
- Data files in `../data/processed/operations_cleaned.csv`

### 1. Start MLflow tracking server
```bash
cd src
docker compose up -d mlflow
```
Wait for MLflow to be ready: http://localhost:5000

### 2. Train the model
```bash
docker compose run --rm model
```
This will:
- Load data from `../data/processed/operations_cleaned.csv`
- Build weekly features (lags, rolling means, multi-step targets)
- Train Random Forest with time-aware split
- Log params/metrics/model to MLflow
- Save artifact to `../models/cement_demand_rf.pkl`

### 3. Start API + Dashboard
```bash
docker compose up -d api streamlit
```

### 4. Access services
- **Dashboard**: http://localhost:8501
- **API docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000

### 5. Test the API
```bash
# Health check
curl http://localhost:8000/health

# List available sites
curl http://localhost:8000/sites

# Forecast
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "site_id": "SITE_001",
    "start_date": "2024-12-31",
    "horizon": 8
  }'
```

## Local Development (without Docker)

### Install dependencies
```bash
cd src
pip install -r requirements.txt
```

### Set environment variables
```bash
export PYTHONPATH=/path/to/Cement-Demand-Forecasting/src
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT=cement_demand_forecasting
```

### Start MLflow locally
```bash
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow/mlflow.db \
  --default-artifact-root file:///mlflow/artifacts
```

### Train model
```bash
python -m src.pipeline \
  --cutoff-date 2022-12-31 \
  --output ../models/cement_demand_rf.pkl \
  --run-name "local-training"
```

### Run API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run Dashboard
```bash
streamlit run src/streamlit/app.py --server.port 8501
```

## Project Structure

```
src/
├── api/
│   └── main.py              # FastAPI app
├── common/
│   ├── config.py            # Centralized settings (env-based)
│   ├── logging_config.py
│   └── schemas.py           # Pydantic request/response models
├── streamlit/
│   └── app.py               # Dashboard
├── data_loader.py           # Load raw/processed CSV
├── preprocessing.py         # Weekly agg, lag/rolling, targets
├── inference.py             # ForecastService (from-date + recursive)
├── pipeline.py              # Training CLI with MLflow
├── requirements.txt
├── Dockerfile.api
├── Dockerfile.model
├── Dockerfile.streamlit
├── docker-compose.yml
└── .env.example
```

## Configuration

All settings are in `src/common/config.py` and can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_ROOT` | `/app` | Root path inside container |
| `DATA_PROCESSED` | `/app/data/processed` | Processed data directory |
| `MODEL_DIR` | `/app/models` | Model artifact directory |
| `MLFLOW_TRACKING_URI` | — | MLflow server URI |
| `MLFLOW_EXPERIMENT` | `cement_demand_forecasting` | Experiment name |
| `API_PORT` | `8000` | API port |
| `DASHBOARD_PORT` | `8501` | Streamlit port |
| `HORIZON` | `8` | Forecast horizon (weeks) |
| `LOG_LEVEL` | `INFO` | Logging level |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health + model loaded status |
| `GET` | `/sites` | List available site IDs |
| `POST` | `/forecast` | Generate forecast |

### Forecast Request
```json
{
  "site_id": "SITE_001",
  "start_date": "2024-12-31",
  "horizon": 8,
  "scenario": [
    {"date": "2025-01-07", "planned_pour_tonnes": 100, "rain_mm": 5, "avg_temp_c": 22}
  ],
  "feature_overrides": {
    "consumed_tonnes_lag_1": 95.5
  }
}
```

### Forecast Response
```json
{
  "site_id": "SITE_001",
  "horizon": 8,
  "forecasts": [
    {"site_id": "SITE_001", "date": "2025-01-07T00:00:00", "forecast_consumed_tonnes": 98.3}
  ],
  "generated_at": "2025-01-01T12:00:00"
}
```

## MLflow Integration

Each training run logs:
- **Parameters**: cutoff_date, n_estimators, max_depth, random_state, horizon, train/test sizes, feature counts
- **Metrics**: MAE/MAPE per horizon (t+1...t+8) + averages
- **Model**: Full sklearn Pipeline with signature & input example
- **Artifact**: Local `.pkl` file

View runs at http://localhost:5000

Load a model from MLflow:
```python
import mlflow
model = mlflow.sklearn.load_model("runs:/<run_id>/model")
```

## CI/CD Notes

Each service has its own Dockerfile for independent builds:
- `Dockerfile.model` — training job (run in CI pipeline)
- `Dockerfile.api` — inference service (deploy to staging/prod)
- `Dockerfile.streamlit` — dashboard (deploy to staging/prod)

Example GitHub Actions workflow:
```yaml
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build model image
        run: docker build -f src/Dockerfile.model -t cement-model .
      - name: Run training
        run: docker run --rm -v ${{ github.workspace }}/models:/app/models cement-model

  build-api:
    needs: train
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build API image
        run: docker build -f src/Dockerfile.api -t cement-api .
      - name: Push to registry
        run: docker push myregistry/cement-api:${{ github.sha }}
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Model not found` | Run `docker compose run --rm model` first |
| `API returns 503` | Check `docker compose logs api` — model may not be loaded |
| `MLflow connection refused` | Ensure `mlflow` service is healthy: `docker compose ps` |
| `No sites in dropdown` | Verify data file exists at `../data/processed/operations_cleaned.csv` |
| `Forecast fails` | Check API logs: `docker compose logs -f api` |

## Data Requirements

The system expects a CSV at `../data/processed/operations_cleaned.csv` with columns:
- `date`, `site_id`, `consumed_tonnes`, `planned_pour_tonnes`
- `rain_mm`, `avg_temp_c`, `silo_capacity`
- `behavior`, `cement_type`, `region`

The preprocessing pipeline handles:
- Daily → weekly aggregation per site
- Lag features (1, 2, 4, 8 weeks)
- Rolling means (4, 8 weeks)
- Multi-step targets (t+1 ... t+8)
src/
├── .dockerignore
├── .env.example
├── data_loader.py          # Load raw/processed CSV, normalize columns
├── docker-compose.yml      # Orchestrates model → api → streamlit
├── Dockerfile.api          # FastAPI inference service (port 8000)
├── Dockerfile.model        # Training pipeline (builds model artifact)
├── Dockerfile.streamlit    # Streamlit dashboard (port 8501)
├── inference.py            # ForecastService: from-date + recursive future
├── pipeline.py             # Training CLI: python -m src.pipeline
├── preprocessing.py        # Weekly agg, lag/rolling, multi-step targets
├── requirements.txt
├── api/
│   ├── __init__.py
│   └── main.py             # FastAPI app: /health, /sites, /forecast
├── common/
│   ├── __init__.py
│   ├── config.py           # Centralized env-based settings
│   ├── logging_config.py
│   └── schemas.py          # Pydantic request/response models
├── model/
│   ├── __init__.py
│   └── cement_demand_rf.pkl  # (artifact, mounted at runtime)
└── streamlit/
    ├── __init__.py
    └── app.py              # Dashboard calling API





Quick start
    # From project root
cd src

# 1. Build & train model (writes to ../models/)
docker compose build model
docker compose run --rm model

# 2. Start API + Streamlit
docker compose up -d api streamlit

# 3. Open dashboard
# http://localhost:8501

# 4. Test API directly
curl http://localhost:8000/health
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"site_id":"SITE_001","start_date":"2024-12-31","horizon":8}'





  Run MLFlow
  # Start MLflow + train
cd src
docker compose up -d mlflow
docker compose run --rm model

# View MLflow UI
# http://localhost:5000

# Or run locally with MLflow
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m src.pipeline --run-name "local-experiment"
"""
Model training pipeline.

Trains a Random Forest model on the engineered features and saves
the pipeline (preprocessing + model) to disk.
"""
import argparse
import os
from pathlib import Path
from typing import Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.common.config import settings
from src.common.logging_config import get_logger
from src.data_loader import load_or_generate_demo
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    NUMERIC_FEATURES,
    build_features,
    get_target_cols,
)

logger = get_logger(__name__)


def setup_mlflow() -> None:
    """Configure MLflow tracking."""
    if settings.MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT)


def build_pipeline(
    n_estimators: int = 300,
    max_depth: int = None,
    random_state: int = 42,
) -> Pipeline:
    """Build the preprocessing + Random Forest pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("rf", rf)])


def time_aware_split(
    model_df: pd.DataFrame, cutoff_date: str = "2022-12-31"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by date cutoff."""
    train_mask = model_df["date"] <= cutoff_date
    test_mask = model_df["date"] > cutoff_date
    X_train = model_df.loc[train_mask, FEATURE_COLS]
    X_test = model_df.loc[test_mask, FEATURE_COLS]
    y_train = model_df.loc[train_mask, get_target_cols()]
    y_test = model_df.loc[test_mask, get_target_cols()]
    return X_train, X_test, y_train, y_test


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    """Compute MAE and MAPE per horizon."""
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, index=y_test.index, columns=y_test.columns)

    metrics = {"mae": {}, "mape": {}}
    for col in y_test.columns:
        mae = mean_absolute_error(y_test[col], y_pred_df[col])
        mape = mean_absolute_percentage_error(y_test[col], y_pred_df[col])
        h = col.split("_")[-1]
        metrics["mae"][h] = float(mae)
        metrics["mape"][h] = float(mape)
    return metrics


def train(
    cutoff_date: str = "2022-12-31",
    output_path: Path = None,
    n_estimators: int = 300,
    max_depth: int = None,
    random_state: int = 42,
    run_name: str = None,
) -> Pipeline:
    """Train the model end-to-end and save it with MLflow tracking."""
    output_path = output_path or settings.MODEL_PATH

    # Setup MLflow
    setup_mlflow()

    logger.info("Loading data...")
    df = load_or_generate_demo()
    weekly, model_df = build_features(df)

    logger.info("Splitting by cutoff %s", cutoff_date)
    X_train, X_test, y_train, y_test = time_aware_split(model_df, cutoff_date)

    logger.info("Training pipeline (n_estimators=%d)...", n_estimators)
    model = build_pipeline(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    # MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_params({
            "cutoff_date": cutoff_date,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
            "horizon": settings.HORIZON,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_features": len(FEATURE_COLS),
            "categorical_features": len(CATEGORICAL_FEATURES),
            "numeric_features": len(NUMERIC_FEATURES),
        })

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate(model, X_test, y_test)

        # Log metrics
        for h, mae in metrics["mae"].items():
            mlflow.log_metric(f"mae_t{h}", mae)
            mlflow.log_metric(f"mape_t{h}", metrics["mape"][h])
            logger.info("t+%s MAE=%.3f MAPE=%.3f", h, mae, metrics["mape"][h])

        # Log overall metrics
        avg_mae = np.mean(list(metrics["mae"].values()))
        avg_mape = np.mean(list(metrics["mape"].values()))
        mlflow.log_metric("avg_mae", avg_mae)
        mlflow.log_metric("avg_mape", avg_mape)

        # Log model with signature
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.iloc[:5],
        )

        # Save locally as well
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
        logger.info("Model saved to %s", output_path)

        # Log artifact
        mlflow.log_artifact(str(output_path), artifact_path="model_artifact")

        logger.info("MLflow run_id: %s", run.info.run_id)

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cement demand forecasting model")
    parser.add_argument("--cutoff-date", default="2022-12-31")
    parser.add_argument("--output", default=str(settings.MODEL_PATH))
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--run-name", default=None, help="MLflow run name")
    args = parser.parse_args()

    train(
        cutoff_date=args.cutoff_date,
        output_path=Path(args.output),
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()

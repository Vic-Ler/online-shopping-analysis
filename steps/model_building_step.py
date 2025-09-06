import logging
from typing import Annotated

import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from zenml import ArtifactConfig, step, Model
from zenml.client import Client
from src.model_building import (
    ModelBuilder,
    RandomForestStrategy,
    XGBoostStrategy,
    LogisticRegressionStrategy,
)

# ZenML experiment tracker + model metadata
experiment_tracker = Client().active_stack.experiment_tracker
model = Model(
    name="revenue_predictor",
    version=None,
    license="Apache 2.0",
    description="Logistic regression model for online session revenue.",
)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str = "logistic",
) -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)]:
    """Train a model using the selected strategy with MLflow logging."""

    # Validate input types
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    # Map strategy string to strategy class
    strategy_map = {
        "rf": RandomForestStrategy(),
        "xgboost": XGBoostStrategy(),
        "logistic": LogisticRegressionStrategy(),
    }

    if strategy not in strategy_map:
        raise ValueError(f"Unknown strategy '{strategy}'. Available: {list(strategy_map.keys())}")

    chosen_strategy = strategy_map[strategy]
    builder = ModelBuilder(chosen_strategy)

    logging.info(f"Training model with '{strategy}' strategy...")

    # Start MLflow run and enable autologging
    if not mlflow.active_run():
        mlflow.start_run()

    try:
        mlflow.sklearn.autolog()

        pipeline = builder.build_and_train(X_train, y_train)

        categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
        numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns
        mlflow.log_param("numerical_columns", list(numerical_cols))
        mlflow.log_param("categorical_columns", list(categorical_cols))

        logging.info("Training complete and MLflow logging finished.")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    finally:
        mlflow.end_run()

    return pipeline

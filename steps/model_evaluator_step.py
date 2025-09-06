import logging
from typing import Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from src.model_evaluator import (
    ModelEvaluator,
    RegressionModelEvaluationStrategy,
    ClassificationModelEvaluationStrategy
)
from zenml import step


@step(enable_cache=False)
def model_evaluator_step(
    trained_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    strategy: str = "regression",   # choose between "regression", "classification"
    plot: bool = True         
) -> Tuple[dict, float]:
    """
    Evaluates the trained model using ModelEvaluator with a chosen strategy.

    Parameters:
    trained_model (Pipeline): The trained pipeline including preprocessing and the model.
    X_test (pd.DataFrame): The test data features.
    y_test (pd.Series): The test data labels/target.
    strategy (str): The evaluation strategy to use ("regression", "classification").

    Returns:
    Tuple[dict, float]: A dictionary with evaluation metrics and a main score (MSE or Accuracy).
    """
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")

    logging.info(f"Evaluating model with strategy: {strategy}")

    # Choose strategy
    if strategy == "regression":
        evaluator = ModelEvaluator(strategy=RegressionModelEvaluationStrategy())
        evaluation_metrics = evaluator.evaluate(trained_model, X_test, y_test, plot=plot)
    elif strategy == "classification":
        evaluator = ModelEvaluator(strategy=ClassificationModelEvaluationStrategy())
        evaluation_metrics = evaluator.evaluate(trained_model, X_test, y_test)
    else:
        raise ValueError(f"Unsupported evaluation strategy: {strategy}")

    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must be returned as a dictionary.")

    # Pick a main metric depending on strategy
    if strategy == "regression":
        main_metric = evaluation_metrics.get("MSE", None)
    elif strategy == "classification":
        main_metric = evaluation_metrics.get("Accuracy", None)
    elif strategy == "crossval":
        main_metric = evaluation_metrics.get("CV Mean Score", None)
    else:
        main_metric = None

    return evaluation_metrics, main_metric

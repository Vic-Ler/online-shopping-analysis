import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.metrics import mean_squared_error, r2_score

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Model Evaluation Strategy
# ----------------------------------------
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        """
        Abstract method to evaluate a model.

        Parameters:
        model (RegressorMixin): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing evaluation metrics.
        """
        pass


# Concrete Strategy for Regression Model Evaluation (MSE + R² already implemented)
# ----------------------------------------
class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(
        self, 
        model: RegressorMixin, 
        X_test: pd.DataFrame, 
        y_test: pd.Series, 
        plot: bool = True  # new flag to show plots
    ) -> dict:
        logging.info("Predicting using the trained regression model.")
        y_pred = model.predict(X_test)

        logging.info("Calculating regression metrics.")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)

        metrics = {"MSE": mse, "RMSE": rmse, "R2": r2}
        logging.info(f"Regression Evaluation Metrics: {metrics}")

        if plot:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Predicted vs True
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=y_test, y=y_pred)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
            plt.xlabel("True Values")
            plt.ylabel("Predicted Values")
            plt.title("Predicted vs True Values")
            plt.show()

            # Residuals
            residuals = y_test - y_pred
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=y_pred, y=residuals)
            plt.axhline(0, color="r", linestyle="--")
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")
            plt.title("Residuals vs Predicted Values")
            plt.show()

            # Histogram of Residuals
            plt.figure(figsize=(8, 6))
            sns.histplot(residuals, kde=True)
            plt.xlabel("Residuals")
            plt.title("Residuals Distribution")
            plt.show()

        return metrics


# Concrete Strategy for Classification Model Evaluation (Accuracy, Precision, Recall, F1)
# ----------------------------------------
class ClassificationModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(
        self, model, X_test: pd.DataFrame, y_test: pd.Series, plot: bool = True
    ) -> dict:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

        logging.info("Predicting using the trained classification model.")
        y_pred = model.predict(X_test)

        logging.info("Calculating classification metrics.")
        cm = confusion_matrix(y_test, y_pred)
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "Confusion Matrix": cm.tolist(),  # store as list for JSON compatibility
        }

        logging.info(f"Classification Evaluation Metrics: {metrics}")

        if plot:
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            plt.show()

        return metrics

# Context Class for Model Evaluation
# ----------------------------------------
class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        """
        Initializes the ModelEvaluator with a specific model evaluation strategy.

        Parameters:
        strategy (ModelEvaluationStrategy): The strategy to be used for model evaluation.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        """
        Sets a new strategy for the ModelEvaluator.

        Parameters:
        strategy (ModelEvaluationStrategy): The new strategy to be used for model evaluation.
        """
        logging.info("Switching model evaluation strategy.")
        self._strategy = strategy

    def evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series, plot: bool = False) -> dict:
        """
        Executes the model evaluation using the current strategy.

        Parameters:
        model (RegressorMixin): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.
        plot (bool): Whether to show plots (only applied if the strategy supports it).

        Returns:
        dict: A dictionary containing evaluation metrics.
        """
        logging.info("Evaluating the model using the selected strategy.")

        # Forward plot=True only if strategy is regression
        if isinstance(self._strategy, RegressionModelEvaluationStrategy):
            return self._strategy.evaluate_model(model, X_test, y_test, plot=plot)
        else:
            return self._strategy.evaluate_model(model, X_test, y_test)
import logging
from abc import ABC, abstractmethod
from typing import Any, Annotated

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Model Building Strategy
# ----------------------------------------
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_pipeline(self, categorical_cols, numerical_cols) -> Pipeline:
        """Return a scikit-learn pipeline with preprocessing + model."""


# Concrete Strategy for Logistic Regression
# ----------------------------------------
class LogisticRegressionStrategy:
    """Strategy for building a Logistic Regression pipeline."""
    
    def build_pipeline(self, categorical_cols, numerical_cols) -> Pipeline:
        logging.info("Using Logistic Regression Strategy.")
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols)
            ]
        )
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(solver="liblinear")) 
            ]
        )
        return pipeline
    
# Concrete Strategy for Random Forest
# ----------------------------------------
class RandomForestStrategy(ModelBuildingStrategy):
    def build_pipeline(self, categorical_cols, numerical_cols) -> Pipeline:
        logging.info("Using RandomForest Classification Strategy.")
        return Pipeline(
            steps=[
                ("preprocessor", ColumnTransformer(
                    transformers=[
                        ("num", SimpleImputer(strategy="mean"), numerical_cols),
                        ("cat", Pipeline([
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]), categorical_cols),
                    ]
                )),
                ("model", RandomForestClassifier(n_estimators=100, random_state=42))
            ]
        )

# Concrete Strategy for XGBoost
# ----------------------------------------
class XGBoostStrategy(ModelBuildingStrategy):
    def build_pipeline(self, categorical_cols, numerical_cols) -> Pipeline:
        logging.info("Using XGBoost Strategy.")
        return Pipeline(
            steps=[
                ("preprocessor", ColumnTransformer(
                    transformers=[
                        ("num", SimpleImputer(strategy="mean"), numerical_cols),
                        ("cat", Pipeline([
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]), categorical_cols),
                    ]
                )),
                ("model", XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                ))
            ]
        )

# Context Class 
# ----------------------------------------
class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        logging.info("Switching model building strategy.")
        self._strategy = strategy

    def build_and_train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
        numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

        logging.info(f"Categorical columns: {categorical_cols.tolist()}")
        logging.info(f"Numerical columns: {numerical_cols.tolist()}")

        pipeline = self._strategy.build_pipeline(categorical_cols, numerical_cols)

        logging.info("Fitting the model pipeline...")
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed.")

        return pipeline
# ZENML WONT WORK ON STREAMLIT
from steps.feature_engineering_step import feature_engineering_step
from steps.outlier_detection_step import outlier_detection_step
from steps.model_loader import load_latest_pipeline
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def predict_from_raw_row(raw_row: pd.DataFrame):
    pipeline = load_latest_pipeline()
    logging.info("Raw row received for prediction:")
    logging.info(raw_row)

    # Apply feature engineering normally, no ZenML
    engineered_row_01 = feature_engineering_step(
        raw_row, strategy="binary_encoding", features=["Revenue", "Weekend"]
    )
    engineered_row_02 = feature_engineering_step(
        engineered_row_01, strategy="month_season_encoding", features=["Month"]
    )
    engineered_row_03 = feature_engineering_step(
        engineered_row_02, strategy="onehot_encoding", features=["Month", "VisitorType"]
    )
    engineered_row_04 = feature_engineering_step(
        engineered_row_03,
        strategy="log",
        features=[
            "ProductRelated_Duration",
            "ProductRelated",
            "Informational_Duration",
            "Informational",
            "Administrative_Duration",
            "Administrative",
            "PageValues",
        ],
    )
    engineered_row_05 = outlier_detection_step(
        engineered_row_04,
        strategy="zscore",
        features=[
            "Informational",
            "Informational_Duration",
            "Administrative",
            "Administrative_Duration",
            "ProductRelated",
            "ProductRelated_Duration",
            "PageValues",
            "BounceRates",
        ],
    )

    logging.info("Engineered row ready:")
    logging.info(engineered_row_05)

    # Align with model features
    if hasattr(pipeline, "feature_names_in_"):
        model_features = pipeline.feature_names_in_
        engineered_row_05 = engineered_row_05.reindex(
            columns=model_features, fill_value=0
        )
        logging.info("Row reindexed to match model features:")
        logging.info(engineered_row_05)

    prediction = pipeline.predict(engineered_row_05)

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(engineered_row_05)
    else:
        probabilities = None

    return prediction, probabilities

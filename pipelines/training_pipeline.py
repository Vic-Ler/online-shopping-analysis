import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitter_step import data_splitter_step
from steps.feature_engineering_step import feature_engineering_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from steps.outlier_detection_step import outlier_detection_step
from zenml import Model, pipeline, step


@pipeline(
    model=Model(
        # unique model name
        name="conversion_predictor"
    ),
)
def ml_pipeline():
    """end-to-end machine learning pipeline"""

    # Data Ingestion Step
    raw_data = data_ingestion_step(
        file_path="c:/Users/Lenovo/Desktop/online-shopping-analysis/data/data.csv"
    )

    # Handling Missing Values Step not needed

    # Feature Engineering binary
    engineered_data_01 = feature_engineering_step(
        raw_data, strategy="binary_encoding", features=["Revenue", "Weekend"]
    )

    # Feature Engineering month to season integer
    engineered_data_02 = feature_engineering_step(
        engineered_data_01, strategy="month_season_encoding", features=["Month"]
    )

    # Feature Engineering one-hot
    engineered_data_03 = feature_engineering_step(
        engineered_data_02, strategy="onehot_encoding", features=["Month", "VisitorType"]
    )

    # Feature Engineering log scaling
    engineered_data_04 = feature_engineering_step(
        engineered_data_03, strategy="log", 
        features=["ProductRelated_Duration", "ProductRelated", "Informational_Duration", "Informational", "Administrative_Duration", "Administrative", "PageValues"]
    )

    # Outlier Detection Step - zscore 
    outliers_detected_05 = outlier_detection_step(
       engineered_data_04, strategy = "zscore", 
        features = ["Informational", "Informational_Duration", "Administrative", "Administrative_Duration", "ProductRelated", "ProductRelated_Duration", "PageValues", "BounceRates"]
    ) # very high threshold since it's skewed and I don't want to loose too much data 
    #Bounce Rates loose the most data (700 instances) but I think it should be fine, as ExitRates correlates with BounceRates

    # Data Splitting Step
    X_train, X_test, y_train, y_test = data_splitter_step(outliers_detected_05, target_column="Revenue", 
                                                          exclude_columns = ["Region", "TrafficType", "Browser", "OperatingSystems"])
    # want to keep dimensionality low if possible

     # Model Building Step
    model = model_building_step(
        X_train=X_train, y_train=y_train, strategy="rf"
    )

    # Model Evaluation Step
    evaluation_metrics = model_evaluator_step(
        trained_model=model,
        X_test=X_test,
        y_test=y_test,
        strategy="classification",
        plot=True
    )

    return model, evaluation_metrics

if __name__ == "__main__":
    run = ml_pipeline()

#python run_pipeline.py
#zenml login --local --blocking
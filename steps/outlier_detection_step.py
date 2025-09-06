import logging
import pandas as pd
from src.outlier_detection import OutlierDetector, ZScoreOutlierDetection, IQROutlierDetection

# from zenml import step  # Commented out for Streamlit

# @step  # Commented out for Streamlit
def outlier_detection_step(df: pd.DataFrame, strategy: str = "zscore", features=None) -> pd.DataFrame:
    """
    Detects and removes outliers using OutlierDetector on one or multiple columns.

    Parameters:
    df (pd.DataFrame): Input dataframe
    strategy (str): "zscore" or "iqr"
    features (str or list): Column name or list of column names to detect outliers

    Returns:
    pd.DataFrame: DataFrame with outliers removed for the specified columns
    """

    logging.info(f"Starting outlier detection step with DataFrame of shape: {df.shape}")

    if df is None:
        logging.error("Received a NoneType DataFrame.")
        raise ValueError("Input df must be a non-null pandas DataFrame.")

    if not isinstance(df, pd.DataFrame):
        logging.error(f"Expected pandas DataFrame, got {type(df)} instead.")
        raise ValueError("Input df must be a pandas DataFrame.")

    if features is None:
        raise ValueError("Please specify at least one feature/column to detect outliers.")

    if isinstance(features, str):
        features = [features]
    elif not isinstance(features, list):
        raise ValueError("features must be a string or a list of strings.")

    # Choose strategy
    if strategy == "zscore":
        outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3.5))
    elif strategy == "iqr":
        outlier_detector = OutlierDetector(IQROutlierDetection())
    else:
        raise ValueError(f"Unsupported outlier detection strategy: {strategy}")

    df_cleaned = df.copy()

    # Detect and handle outliers for all specified columns at once
    outliers = outlier_detector.detect_outliers(df_cleaned, features=features)
    df_cleaned = outlier_detector.handle_outliers(df_cleaned, features=features, method="remove")

    logging.info(f"Outlier detection completed. Cleaned DataFrame shape: {df_cleaned.shape}")
    return df_cleaned

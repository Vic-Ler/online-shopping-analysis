import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Outlier Detection Strategy
# ----------------------------------------
class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        Abstract method to detect outliers in the given DataFrame for specific features.

        Parameters:
        df (pd.DataFrame): The dataframe containing features for outlier detection.
        features (list): The list of features/columns to analyze for outliers.

        Returns:
        pd.DataFrame: A boolean dataframe indicating where outliers are located for the specified features.
        """
        pass

# Concrete Strategy for Z-Score Based Outlier Detection
# ----------------------------------------
class ZScoreOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, threshold=3.5):
        self.threshold = threshold

    def detect_outliers(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        logging.info("Detecting outliers using the Z-score method.")
        z_scores = np.abs((df[features] - df[features].mean()) / df[features].std())
        outliers = z_scores > self.threshold
        
        # Count outliers per feature
        outlier_counts = outliers.sum()
        for feature, count in outlier_counts.items():
            logging.info(f"Feature '{feature}': {count} outliers detected.")

        logging.info(f"Outliers detected with Z-score threshold: {self.threshold}.")
        return outliers


# Concrete Strategy for IQR Based Outlier Detection
# ----------------------------------------
class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        logging.info("Detecting outliers using the IQR method.")
        Q1 = df[features].quantile(0.25)
        Q3 = df[features].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df[features] < (Q1 - 1.5 * IQR)) | (df[features] > (Q3 + 1.5 * IQR))
        
        # Count outliers per feature
        outlier_counts = outliers.sum()
        for feature, count in outlier_counts.items():
            logging.info(f"Feature '{feature}': {count} outliers detected.")
        
        logging.info("Outliers detected using the IQR method.")
        return outliers

# Context Class for Outlier Detection and Handling
# ----------------------------------------
class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy):
        logging.info("Switching outlier detection strategy.")
        self._strategy = strategy

    def detect_outliers(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        logging.info("Executing outlier detection strategy.")
        return self._strategy.detect_outliers(df, features)

    def handle_outliers(self, df: pd.DataFrame, features: list, method="remove", **kwargs) -> pd.DataFrame:
        outliers = self.detect_outliers(df, features)
        if method == "remove":
            logging.info("Removing outliers from the dataset.")
            df_cleaned = df[(~outliers).all(axis=1)]
        elif method == "cap":
            logging.info("Capping outliers in the dataset.")
            df_cleaned = df.copy()
            for feature in features:
                lower = df[feature].quantile(0.01)
                upper = df[feature].quantile(0.99)
                df_cleaned[feature] = df[feature].clip(lower=lower, upper=upper)
        else:
            logging.warning(f"Unknown method '{method}'. No outlier handling performed.")
            return df

        logging.info("Outlier handling completed.")
        return df_cleaned

    def visualize_outliers(self, df: pd.DataFrame, features: list):
        logging.info(f"Visualizing outliers for features: {features}")
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
        logging.info("Outlier visualization completed.")

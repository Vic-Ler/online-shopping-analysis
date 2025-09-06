import logging 
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Feature Engineering Strategy
# ----------------------------------------------------
# This class defines a common interface for different feature engineering strategies.
# Subclasses must implement the apply_transformation method.
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame, features: list = None) -> pd.DataFrame:
        """
        Abstract method to apply feature engineering transformation to the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.
        features (list, optional): The list of features to transform. If None, use default features.

        Returns:
        pd.DataFrame: A dataframe with the applied transformations.
        """
        pass
 

# Concrete Strategy for Log Transformation
# ----------------------------------------
# This strategy applies a logarithmic transformation to skewed features to normalize the distribution.
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the LogTransformation with the specific features to transform.

        Parameters:
        features (list): The list of features to apply the log transformation to.
        """
        self.features = features

    def apply_transformation(self, df: pd.DataFrame, features: list = None) -> pd.DataFrame:
        """
        Applies a log transformation to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.
        features (list, optional): The list of features to transform. Overrides default features.

        Returns:
        pd.DataFrame: The dataframe with log-transformed features.
        """
        features_to_use = features if features is not None else self.features
        logging.info(f"Applying log transformation to features: {features_to_use}")
        df_transformed = df.copy()
        for feature in features_to_use:
            df_transformed[feature] = np.log1p(df_transformed[feature])  # log1p handles log(0)
        logging.info("Log transformation completed.")
        return df_transformed


# Concrete Strategy for Standard Scaling
# --------------------------------------
# This strategy applies standard scaling (z-score normalization) to features, centering them around zero with unit variance.
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the StandardScaling with the specific features to scale.

        Parameters:
        features (list): The list of features to apply the standard scaling to.
        """
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame, features: list = None) -> pd.DataFrame:
        """
        Applies standard scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.
        features (list, optional): The list of features to transform. Overrides default features.

        Returns:
        pd.DataFrame: The dataframe with scaled features.
        """
        features_to_use = features if features is not None else self.features
        logging.info(f"Applying standard scaling to features: {features_to_use}")
        df_transformed = df.copy()
        df_transformed[features_to_use] = self.scaler.fit_transform(df_transformed[features_to_use])
        logging.info("Standard scaling completed.")
        return df_transformed


# Concrete Strategy for Min-Max Scaling
# -------------------------------------
# This strategy applies Min-Max scaling to features, scaling them to a specified range, typically [0, 1].
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features, feature_range=(0, 1)):
        """
        Initializes the MinMaxScaling with the specific features to scale and the target range.

        Parameters:
        features (list): The list of features to apply the Min-Max scaling to.
        feature_range (tuple): The target range for scaling, default is (0, 1).
        """
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame, features: list = None) -> pd.DataFrame:
        """
        Applies Min-Max scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.
        features (list, optional): The list of features to transform. Overrides default features.

        Returns:
        pd.DataFrame: The dataframe with Min-Max scaled features.
        """
        features_to_use = features if features is not None else self.features
        logging.info(
            f"Applying Min-Max scaling to features: {features_to_use} with range {self.scaler.feature_range}"
        )
        df_transformed = df.copy()
        df_transformed[features_to_use] = self.scaler.fit_transform(df_transformed[features_to_use])
        logging.info("Min-Max scaling completed.")
        return df_transformed


# Concrete Strategy for One-Hot Encoding
# --------------------------------------
# This strategy applies one-hot encoding to categorical features, converting them into binary vectors.
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the OneHotEncoding with the specific features to encode.

        Parameters:
        features (list): The list of categorical features to apply the one-hot encoding to.
        """
        self.features = features
        self.encoder = OneHotEncoder(sparse_output=False, drop="first")

    def apply_transformation(self, df: pd.DataFrame, features: list = None) -> pd.DataFrame:
        """
        Applies one-hot encoding to the specified categorical features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.
        features (list, optional): The list of features to transform. Overrides default features.

        Returns:
        pd.DataFrame: The dataframe with one-hot encoded features.
        """
        features_to_use = features if features is not None else self.features
        logging.info(f"Applying one-hot encoding to features: {features_to_use}")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df_transformed[features_to_use]),
            columns=self.encoder.get_feature_names_out(features_to_use),
        )
        df_transformed = df_transformed.drop(columns=features_to_use).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("One-hot encoding completed.")
        return df_transformed

# Concrete Strategy for Binary Encoding
# --------------------------------------
# This strategy applies binary encoding to boolean features, converting them into binary.
class BooleanToBinaryEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the BooleanToBinaryEncoding with the specific features to transform.

        Parameters:
        features (list): The list of boolean features to convert into binary (0/1).
        """
        self.features = features

    def apply_transformation(self, df: pd.DataFrame, features: list = None) -> pd.DataFrame:
        """
        Converts specified boolean features in the DataFrame into binary (0/1).

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.
        features (list, optional): The list of features to transform. Overrides default features.

        Returns:
        pd.DataFrame: The dataframe with boolean features transformed into binary.
        """
        features_to_use = features if features is not None else self.features
        logging.info(f"Converting boolean features {features_to_use} to binary (0/1).")
        df_transformed = df.copy()
        for feature in features_to_use:
            if feature in df_transformed.columns:
                df_transformed[feature] = df_transformed[feature].astype(int)
            else:
                logging.warning(f"Feature '{feature}' not found in DataFrame. Skipping.")
        logging.info("Boolean to binary conversion completed.")
        return df_transformed

# Concrete Strategy for Categorical to Integer Encoding
# ------------------------------------------------------
# This strategy transforms categorical string features into integer codes.
class CategoricalToIntegerEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the CategoricalToIntegerEncoding with the specific features to transform.

        Parameters:
        features (list): The list of categorical features to convert into integers.
        """
        self.features = features

    def apply_transformation(self, df: pd.DataFrame, features: list = None) -> pd.DataFrame:
        """
        Converts specified categorical string features in the DataFrame into integer codes.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.
        features (list, optional): The list of features to transform. Overrides default features.

        Returns:
        pd.DataFrame: The dataframe with categorical features transformed into integer codes.
        """
        features_to_use = features if features is not None else self.features
        logging.info(f"Converting categorical features {features_to_use} to integer codes.")
        df_transformed = df.copy()
        for feature in features_to_use:
            if feature in df_transformed.columns:
                df_transformed[feature] = df_transformed[feature].astype('category').cat.codes
            else:
                logging.warning(f"Feature '{feature}' not found in DataFrame. Skipping.")
        logging.info("Categorical to integer conversion completed.")
        return df_transformed
    
# Conrete Strategy for converting months into season integers
# -------------------------------------
# This strategy transforms labelled months (strings) into season (integers).
class MonthsToSeasonsEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the MonthsToSeasons transformation.

        Parameters:
        features (list): The list of columns containing month strings (e.g., ['month_col1', 'month_col2']).
        """
        self.features = features
        # Map months to seasons (integer codes)
        self.month_to_season = {
            "Dec": 0, "Jan": 0, "Feb": 0,  # Winter
            "Mar": 1, "Apr": 1, "May": 1,  # Spring
            "Jun": 2, "Jul": 2, "Aug": 2,  # Summer
            "Sep": 3, "Oct": 3, "Nov": 3   # Autumn
        }

    def apply_transformation(self, df: pd.DataFrame, features: list = None) -> pd.DataFrame:
        """
        Converts month strings to season integers.

        Parameters:
        df (pd.DataFrame): The dataframe containing the month columns.
        features (list, optional): Overrides the default features list.

        Returns:
        pd.DataFrame: The dataframe with month columns replaced by season codes.
        """
        features_to_use = features if features is not None else self.features
        df_transformed = df.copy()

        for col in features_to_use:
            if col not in df_transformed.columns:
                logging.warning(f"Feature '{col}' not found in DataFrame. Skipping transformation.")
                continue
            logging.info(f"Transforming month column '{col}' to seasons.")
            df_transformed[col] = df_transformed[col].map(self.month_to_season)

        logging.info("Month to season transformation completed.")
        return df_transformed
    
# Context Class for Feature Engineering
# -------------------------------------
# This class uses a FeatureEngineeringStrategy to apply transformations to a dataset.
class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        """
        Initializes the FeatureEngineer with a specific feature engineering strategy.

        Parameters:
        strategy (FeatureEngineeringStrategy): The strategy to be used for feature engineering.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """
        Sets a new strategy for the FeatureEngineer.

        Parameters:
        strategy (FeatureEngineeringStrategy): The new strategy to be used for feature engineering.
        """
        logging.info("Switching feature engineering strategy.")
        self._strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame, features: list = None) -> pd.DataFrame:
        """
        Executes the feature engineering transformation using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.
        features (list, optional): The list of features to transform. Overrides default features.

        Returns:
        pd.DataFrame: The dataframe with applied feature engineering transformations.
        """
        logging.info("Applying feature engineering strategy.")
        return self._strategy.apply_transformation(df, features)

from abc import ABC, abstractmethod
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Abstract Base Class for Univariate Analysis Strategy
# -----------------------------------------------------
# This class defines a common interface for univariate analysis strategies.
# Subclasses must implement the analyze method.
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature=None):
        """
        Perform univariate analysis on a specific feature of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str or list): The name(s) of the feature(s)/column(s) to be analyzed.

        Returns:
        None: This method visualizes the distribution of the feature(s).
        """
        pass


# Concrete Strategy for Numerical Features
# -----------------------------------------
# This strategy analyzes numerical features by plotting their distribution.
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature=None):
        """
        Plots the distribution of (a) numerical feature(s) using a histogram and KDE.
        Default are all detected numerical features.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str or list): The name(s) of the numerical feature(s)/column(s) to be analyzed.

        Returns:
        None: Displays a histogram with a KDE plot.
        """
        # Determine features to analyze
        if feature is None:
            features = df.select_dtypes(include=["int64", "float64", "Int64"]).columns.tolist()
        elif isinstance(feature, str):
            features = [feature]
        else:
            features = feature 

        n_features = len(features)
        if n_features == 0:
            print("No numeric features to analyze.")
            return

        n_cols = 2
        n_rows = math.ceil(n_features / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*4))
        axes = axes.flatten()
        for i, f in enumerate(features):
            sns.histplot(df[f], kde=True, bins=30, ax=axes[i])
            axes[i].set_title(f"Distribution of {f}")
            axes[i].set_xlabel(f)
            axes[i].set_ylabel("Frequency")
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()


# Concrete Strategy for Categorical Features
# -------------------------------------------
# This strategy analyzes categorical features by plotting their frequency distribution.
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature=None):
        """
        Plots the distribution of (a) categorical feature(s) using a bar plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str or list): The name(s) of the categorical feature(s)/column(s) to be analyzed.

        Returns:
        None: Displays a bar plot showing the frequency of each category.
        """
        if feature is None:
            features = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        elif isinstance(feature, str):
            features = [feature]
        else:
            features = feature 

        n_features = len(features)
        if n_features == 0:
            print("No categorical features to analyze.")
            return

        n_cols = 2
        n_rows = math.ceil(n_features / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*4))
        axes = axes.flatten() 
        for i, f in enumerate(features):
            sns.countplot(x=f, data=df, palette="muted", ax=axes[i])
            axes[i].set_title(f"Distribution of {f}")
            axes[i].set_xlabel(f)
            axes[i].set_ylabel("Count")
            axes[i].tick_params(axis='x', rotation=45)
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()


# Context Class that uses a UnivariateAnalysisStrategy
# ----------------------------------------------------
# This class allows you to switch between different univariate analysis strategies.
class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        """
        Initializes the UnivariateAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The strategy to be used for univariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        """
        Sets a new strategy for the UnivariateAnalyzer.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The new strategy to be used for univariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature=None):
        """
        Executes the univariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str or list): The name(s) of the feature(s)/column(s) to be analyzed.

        Returns:
        None: Executes the strategy's analysis method and visualizes the results.
        """
        self._strategy.analyze(df, feature)

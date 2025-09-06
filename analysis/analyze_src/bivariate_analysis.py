from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Abstract Base Class for Bivariate Analysis Strategy
# ----------------------------------------------------
# This class defines a common interface for bivariate analysis strategies.
# Subclasses must implement the analyze method.
class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, target: str = None):
        """
        Perform bivariate analysis on two features of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first feature/column to be analyzed.
        feature2 (str): The name of the second feature/column to be analyzed.
        target (str, optional): The name of the target column to use for coloring or grouping. Default is None.

        Returns:
        None: This method visualizes the relationship between the two features.
        """
        pass


# Concrete Strategy for Numerical vs Numerical Analysis
# ------------------------------------------------------
class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, target: str = None):
        """
        Plots the relationship between two numerical features using a scatter plot.
        Optionally, a target column can be used as hue for coloring the points.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=df, hue=target)
        plt.title(f"{feature1} vs {feature2}" + (f" colored by {target}" if target else ""))
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()


# Concrete Strategy for Categorical vs Numerical Analysis
# --------------------------------------------------------
class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, target: str = None):
        """
        Plots the relationship between a categorical feature and a numerical feature using a box plot.
        The target parameter is ignored for this strategy.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()


# Concrete Strategy for Categorical vs Categorical Analysis (Heatmap)
# -------------------------------------------------------------------
class CategoricalVsCategoricalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, target: str = None):
        """
        Plots the relationship between two categorical features using a heatmap.
        The target parameter is ignored for this strategy.
        """
        contingency_table = pd.crosstab(df[feature1], df[feature2])
        plt.figure(figsize=(10, 6))
        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature2)
        plt.ylabel(feature1)
        plt.show()


# Concrete Strategy for Categorical vs Categorical Analysis using Bar Charts
# --------------------------------------------------------------------------
class CategoricalVsCategoricalBarAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, target: str = None):
        """
        Plots the relationship between two categorical features using a bar chart.
        The x-axis represents the first categorical feature and the bars are colored by the second categorical feature.
        The target parameter is ignored for this strategy.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature1, hue=feature2, data=df, palette='Set2')
        plt.title(f"{feature1} vs {feature2} (Bar Chart)")
        plt.xlabel(feature1)
        plt.ylabel("Count")
        plt.legend(title=feature2)
        plt.show()


# Context Class that uses a BivariateAnalysisStrategy
# ---------------------------------------------------
class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        """
        Initializes the BivariateAnalyzer with a specific analysis strategy.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        """
        Sets a new strategy for the BivariateAnalyzer.
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str, target: str = None):
        """
        Executes the bivariate analysis using the current strategy.
        """
        self._strategy.analyze(df, feature1, feature2, target)

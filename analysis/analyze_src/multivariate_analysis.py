from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Abstract Base Class for Multivariate Analysis
# ----------------------------------------------
# This class defines a template for performing multivariate analysis.
# Subclasses can override specific steps like correlation heatmap and pair plot generation.
class MultivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame):
        """
        Perform a comprehensive multivariate analysis by generating a correlation heatmap and pair plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: This method orchestrates the multivariate analysis process.
        """
        pass


# Concrete Class for Multivariate Analysis with Correlation Heatmap
# -----------------------------------------------------------------
# This class implements the method to generate only a correlation heatmap.
class CorrelationHeatmapAnalysis(MultivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame):
        """
        Generates and displays a correlation heatmap for the numerical features in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: Displays a heatmap showing correlations between numerical features.
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()


# Concrete Class for Multivariate Analysis with Pair Plot
# -------------------------------------------------------
# This class implements the method to generate only a pair plot.
class PairPlotAnalysis(MultivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame):
        """
        Generates and displays a pair plot for the selected features in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: Displays a pair plot for the selected features.
        """
        sns.pairplot(df)
        plt.suptitle("Pair Plot of Selected Features", y=1.02)
        plt.show()


# Context Class that uses a MultivariateAnalysisStrategy
# -------------------------------------------------------
class MultivariateAnalyser:
    def __init__(self, strategy: MultivariateAnalysisStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: MultivariateAnalysisStrategy):
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame):
        self._strategy.analyze(df)


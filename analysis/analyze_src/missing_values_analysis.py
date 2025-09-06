from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Abstract Base Class for Missing Values Analysis
# --------------------------------------------------
# This class defines a common interface for missing values analysis.
# Subclasses must implement the analyse method.
class MissingValuesStrategy(ABC):
    @abstractmethod
    def analyse(self, df: pd.DataFrame):
        """
        Perform a specific type of missing value analysis.
        
        Parameters:
        df (pd.DataFrame): The dataframe on which the analysis is to be performed.

        Returns:
        None: This method prints the analysis results directly.
        """
        pass

# Concrete Strategy for Missing Value Analysis
# --------------------------------------------
# This strategy analysis missing values for all columns and displays them in a heatmap.
class MissingValuesHeatMapStrategy(MissingValuesStrategy):
    def analyse(self, df: pd.DataFrame):
        """
        Prints the count of missing values for each column in the dataframe.
        Creates a heatmap to visualize the missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: Prints the missing values count to the console. Displays a heatmap of missing values.
        """
        print("\nMissing Values Count by Column:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])
        print("\nVisualizing Missing Values...")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()

# Context Class that uses a MissingValuesAnalysisStrategy
# ------------------------------------------------
# This class allows you to switch between different missing values analysis strategies.
class MissingValuesAnalyser:
    def __init__(self, strategy: MissingValuesStrategy):
        """
        Initializes the MissingValuesAnalyser with a specific analysis strategy.

        Parameters:
        strategy (MissingValuesStrategy): The strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValuesStrategy):
        """
        Sets a new strategy for the MissingValuesAnalyser.

        Parameters:
        strategy (MissingValuesStrategy): The new strategy to be used for analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        """
        Executes the inspection using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Executes the strategy's analysis method.
        """
        self._strategy.analyse(df)

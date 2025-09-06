import os
import zipfile
from abc import ABC, abstractmethod

import pandas as pd


# Abstract Base Class for Data Ingestor
# ----------------------------------------------------
# This class defines a common interface for different data ingestion strategies.
# Subclasses must implement the ingest method.
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Abstract method to ingest data from a given file.

        Parameters:
        file_path: str: The file path to the respective file to be ingested.

        Returns:
        pd.DataFrame: A dataframe with the ingested data.
        """
        pass


# Concrete Method for ZIP Ingestion
# ----------------------------------------
# This method applies ingests data formatted as ZIP.
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Extracts a .zip file and returns the content as a pandas DataFrame.

        Parameters:
        file_path: str: The file path to the respective file to be ingested.
        """
        # Ensure the file is a .zip
        if not file_path.endswith(".zip"):
            raise ValueError("The provided file is not a .zip file.")

        # Extract the zip file
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("data")

        # Find the extracted CSV file (assuming there is one CSV file inside the zip)
        extracted_files = os.listdir("data")
        csv_files = [f for f in extracted_files if f.endswith(".csv")]

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the extracted data.")
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV files found. Please specify which one to use.")

        # Read the CSV into a DataFrame
        csv_file_path = os.path.join("data", csv_files[0])
        df = pd.read_csv(csv_file_path)

        # Return the DataFrame
        return df

# Concrete Method for CSV Ingestion
# ----------------------------------------
# This method applies ingests data formatted as CSV.
class CSVDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Reads a CSV file and returns the content as a pandas DataFrame.

        Parameters:
        file_path: str: The file path to the respective file to be ingested.
        """        
        if not file_path.endswith(".csv"):
            raise ValueError("The provided file is not a .csv file.")

        df = pd.read_csv(file_path)
        return df

# Context Class/ Factory for Ingestion
# ----------------------------------------
# This factory defines the method to be used based on input.
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """
        Returns the appropriate DataIngestor based on file extension.

        Parameters:
        file_path: str: The file path to the respective file to be ingested.
        file_extension: str: File extension of respective file to be ingested.
        """
        if file_extension == ".zip":
            return ZipDataIngestor()
        elif file_extension == ".csv":
            return CSVDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")



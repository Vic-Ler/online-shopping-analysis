import os
import logging
from abc import ABC, abstractmethod
import pandas as pd
from ucimlrepo import fetch_ucirepo

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Fetching Strategy
# -----------------------------------------------
# This class defines a common interface for different data fetching strategies.
# Subclasses must implement the fetch_data method.
class DataFetchStrategy(ABC):
    @abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        """
        Abstract method to fetch data.

        Parameters:
        None: the method does not take any arguments; 
        the strategy itself should have any needed info (like dataset ID).

        Returns:
        pd.DataFrame: The fetched dataset as a DataFrame.
        """
        pass

# Concrete Strategy for fetching Data from UCI.
# ---------------------------------------------
# This strategy fetches data from UCI.
class UCIFetchStrategy(DataFetchStrategy):
    def __init__(self, dataset_id: int):
        self.dataset_id = dataset_id

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch dataset from UCI repository using fetch_ucirepo.

        Parameters: 
        ID of the dataset

        Returns: 
        pd.DataFrame: The fetched dataset as a DataFrame.
        """
        logging.info(f"Fetching dataset with ID {self.dataset_id} from UCI Repo...")
        try:
            dataset = fetch_ucirepo(id=self.dataset_id)
            df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
            logging.info(f"Successfully fetched dataset with shape {df.shape}")
            return df
        except Exception as e:
            logging.error("Failed to fetch dataset from UCI repository.", exc_info=True)
            raise RuntimeError("Failed to fetch dataset from UCI repository.") from e
        
# Context Class for Data Fetching
# --------------------------------
# This class uses a DataFetchStrategy to fetch the data.
class DataFetcher:
    """
    Context class that uses a DataFetchStrategy to fetch data and save it to a fixed path
    relative to the project root.
    """

    def __init__(self, strategy: 'DataFetchStrategy'):
        """
        Initializes the DataFetcher with a specific fetching strategy.

        Parameters:
        strategy (DataFetchStrategy): The strategy to be used for data fetching.
        """
        self._strategy = strategy

        current_file = os.path.abspath(__file__)  
        project_root = os.path.dirname(os.path.dirname(current_file))  
        self.DEFAULT_SAVE_PATH = os.path.join(project_root, "data", "data.csv")

    def set_strategy(self, strategy: 'DataFetchStrategy'):
        """
        Change the fetching strategy at runtime.

        Parameters:
        strategy (DataFetchStrategy): The new strategy to be used for data fetching.
        """
        logging.info("Switching data fetching strategy.")
        self._strategy = strategy

    def fetch_and_save(self) -> pd.DataFrame:
        """
        Executes the data fetching using the current strategy and saves it to the default path.

        Returns:
        pd.DataFrame: The fetched dataset as a DataFrame.
        """
        logging.info("Fetching data using the selected strategy.")
        df = self._strategy.fetch_data()

        os.makedirs(os.path.dirname(self.DEFAULT_SAVE_PATH), exist_ok=True)

        df.to_csv(self.DEFAULT_SAVE_PATH, index=False)
        logging.info(f"Data saved to {self.DEFAULT_SAVE_PATH}")

        return df

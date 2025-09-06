from typing import Tuple, List

import pandas as pd
from src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy
from zenml import step


@step
def data_splitter_step(
    df: pd.DataFrame, 
    target_column: str,
    exclude_columns: List[str] = None 
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the data into training and testing sets using DataSplitter and a chosen strategy."""
    
    splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy())
    
    X_train, X_test, y_train, y_test = splitter.split(
        df, target_column=target_column, exclude_columns=exclude_columns
    )
    
    return X_train, X_test, y_train, y_test

import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataPreProcessStrategy, DataSplitterStrategy , DataCleaning
from typing_extensions import Annotated
from typing import Tuple



@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
] :
    

    """
    clean the data and splits it into train and test sets
    Args:
    df : the ingested data
    returns:
    X_train : training features
    X_test : testing features
    y_train : training labels
    y_test : testing labels
    
    """
    try:
        """
        Clean the DataFrame by applying data cleaning strategies.
        """
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df , preprocess_strategy)
        processed_data = data_cleaning.handle_data()

        splitting_strategy = DataSplitterStrategy()
        data_cleaning = DataCleaning(processed_data , splitting_strategy)
        X_train , X_test , y_train , y_test = data_cleaning.handle_data()
        logging.info("Data cleaning and splitting completed successfully.")
        return X_train , X_test , y_train , y_test

    except Exception as e:
        logging.error("Error in cleaning the data: {}".format(e))
        raise e



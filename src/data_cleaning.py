import logging
import pandas as pd
from abc import ABC , abstractmethod
from typing import Union
import numpy as np
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract class for defining strategy for handling data
    
    """


    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        pass


class DataPreProcessStrategy(DataStrategy):
    """
    strategy for preprocessing the data
    
    """

    def handle_data(self, data:pd.DataFrame) -> pd.DataFrame:

        """preprocess the data"""
        
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp"
                ],
                axis=1
            )

            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            return data
        
        except Exception as e:
            logging.error("Error in preprocessing the data: {}".format(e))
            raise e


class DataSplitterStrategy(DataStrategy):
    """
    strategy for splitting the data into train and test sets
    
    """

    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        """
        split the data into train and test sets
        """
        try:
            X = data.drop(["review_score"], axis = 1)
            y = data["review_score"]

            X_train , X_test , y_train , y_test = train_test_split(
                X , y , test_size=0.1 , random_state=42
            )

            return X_train , X_test , y_train , y_test
        
        except Exception as e:
            logging.error("Error in splitting the data: {}".format(e))
            raise e
        

class DataCleaning :
    """
    class for cleaning the data which processes the data and split it into training and testing sets
    """

    def __init__ (self , data:pd.DataFrame , strategy:DataStrategy):
        """
        Args:
            data : the ingested data
            strategy : the strategy for handling the data
        """
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame , pd.Series]:
        """
        handle the data using the strategy
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e

# if __name__ == "__main__":
#     data = pd.read_csv("/home/ayushz/ML/MLOps/Project_1/data/olist_customers_dataset.csv")
#     data_cleaning = DataCleaning(data , DataPreProcessStrategy())
#     data_cleaning.handle_data()
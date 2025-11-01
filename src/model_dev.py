import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    abstract class for all models
    """

    @abstractmethod
    def train(self, X_train , y_train):
        """
        train the model
        
        Args:
            X_train : training features
            y_train : training labels
        
        """
        pass



class LinearRegressionModel(Model):
    """
    Linear Regression Model
    """
    def train(self, X_train , y_train , **kwargs):
        """
        train the Linear Regression model
        
        Args:
            X_train : training features
            y_train : training labels
        
        """
        
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train , y_train)
            logging.info("Linear Regression model trained successfully.")
            return reg
        except Exception as e:
            logging.error("Error in training Linear Regression model: {}".format(e))
            raise e
        

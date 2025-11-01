import logging
from abc import ABC , abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error , r2_score , root_mean_squared_error

class Evaluation(ABC):
    """
    abstract class defining strategy for evaluation our models
    """
    @abstractmethod
    def calculate_scores(self, y_true:np.ndarray , y_pred: np.ndarray):
        """
        calculates the scores for the model
        Args:
        y_true: True labels
        y_pred : predicted labels

        returns:
        None
        
        """

        pass


class MSE(Evaluation):

    """
    Evaluation strategy that uses Mean squared errors

    mse = np.mean((y_true-y_pred)**2)

    """
    def calculate_scores(self, y_true: np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true,y_pred)
            # mse = np.mean((y_true-y_pred)**2)

            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e
        


class RMSE(Evaluation):
    """
    evaluation strategy that uses rmse which is root of the mean squarred error
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("calculating RMSE")
            RMSE = root_mean_squared_error(y_true,y_pred)
            logging.info("RMSE {}".format(RMSE))
            return RMSE
        except Exception as e:
            logging.error("Error in claculating RMSE: {}".format(e))
            raise e

class R2(Evaluation):
    """
    Evaluation Strategy that uses R2 score

    """

    def calculate_scores(self, y_true: np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("Calculating r2")
            r2 = r2_score(y_true,y_pred)

            logging.info("r2: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating r2_score: {}".format(e))
            raise e

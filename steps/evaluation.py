import logging
import pandas as pd
from zenml import step
from src.evaluate import MSE , RMSE , R2
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from typing import Tuple
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
         y_test : pd.DataFrame) -> Tuple[
      Annotated[float, "R2 Score"],
      Annotated[float, "RMSE"],
      Annotated[float, "MSE"]
         ]:

    """
    Evaluate the model on the ingested data,
    Args:
       df: the ingested data
    
    
    """
    try:
         prediction = model.predict(X_test)
         mse_class = MSE()
         mse = mse_class.calculate_scores(y_test , prediction)
         mlflow.log_metric("MSE" , mse)

         r2_class = R2()
         r2 = r2_class.calculate_scores(y_test , prediction)
         mlflow.log_metric("R2" , r2)

         rmse_class = RMSE()
         rmse = rmse_class.calculate_scores(y_test, prediction)
         mlflow.log_metric("RMSE" , rmse)
         
         return r2 , rmse , mse
    except Exception as e:
       logging.error(f"Error occurred while evaluating model: {e}")
    
       
      
   


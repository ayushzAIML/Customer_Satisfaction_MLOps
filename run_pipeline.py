from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="/home/ayushz/ML/MLOps/Project_1/data/olist_customers_dataset.csv")




"""

to check mlflow uri bash the bottom command which can be different after runing each pipeline

*
 mlflow ui --backend-store-uri "file:/home/ayushz/.config/zenml/local_stores/f7319b62-9feb-4aba-9d68-ec087f97d376/mlruns"
*
"""



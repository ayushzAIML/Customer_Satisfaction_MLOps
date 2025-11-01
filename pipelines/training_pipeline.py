from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from steps.config import ModelNameConfig

@pipeline(enable_cache=True)

def train_pipeline(data_path:str):
    """
    The training pipeline that ingests, cleans, trains and evaluates the model.

    Args:
    data_path : path to the data

    """
    df = ingest_df(data_path=data_path)
    X_train , X_test , y_train , y_test = clean_df(df=df)
    model = train_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, config = ModelNameConfig())
    r2 , rmse , mse = evaluate_model(model , X_test = X_test , y_test = y_test)




import numpy as np
import pandas as pd
from zenml import pipeline , step
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
# from materializer.custom_materializers import cs_materializer
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT

from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from pydantic import BaseModel

from steps.clean_data import clean_df
from steps.config import ModelNameConfig
from steps.ingest_data import ingest_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from sklearn.metrics import r2_score
from pipelines.utils import get_data_for_test
import json


docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseModel):
    min_accuracy: float = 0

@step(enable_cache=False)
def dynamic_importer()->str:
    data = get_data_for_test()
    return data


@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig
):
    """implements a simple model deployment trigger that looks at the input model accuracy and
    decides it is good enough to deploy or not."""
    # return accuracy >= config.min_accuracy --> in case of classification
    return accuracy > config.min_accuracy

class MLFlowDeploymentLoaderStepParameters(BaseModel):
    """
    MLflow deployment getter parameters.

    Attributes:
    pipeline_name: The name of the pipeline. that deployed mlflow prediction

    step_name: The name of the step that deployed the mlflow prediction model.

    running: when this flag is set , the step will only return
    a running deployment service.

    model_name: the name of the model that was deployed.
    """

    pipeline_name: str
    step_name: str
    running: bool = True

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model"
) ->MLFlowDeploymentService:
    """A step to load an existing MLflow deployment service from a previous pipeline run.

    Args:
        pipeline_name: The name of the pipeline that deployed the MLflow model.
        pipeline_step_name: The name of the step that deployed the MLflow model.
        running: If True, only return a running deployment service.
        model_name: The name of the deployed model."""
    
    #get mlflow deployer stack component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    #fetch existing services within the pipeline , step name and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service found for pipeline '{pipeline_name}', "
            f"step '{pipeline_step_name}', and model '{model_name}'. "
            f"Please deploy a model first using --config deploy."
        )

    return existing_services[0]


@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,

) -> np.ndarray:
    service.start(timeout=60) #should be a NOP if already started
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df =[
        "payment_sequential",
        "payment_instalments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    predictions = service.predict(df)
    return predictions
    


@pipeline(enable_cache=False , settings={"docker":docker_settings})
def continuous_deployment_pipeline(

    data_path: str ,
    min_accuracy : float = 0,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    df = ingest_df(data_path=data_path)
    X_train , X_test , y_train , y_test = clean_df(df=df)
    model = train_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, config=ModelNameConfig())
    r2 , rmse , mse = evaluate_model(model , X_test = X_test , y_test = y_test)

    deployment_decision = deployment_trigger(accuracy = r2, config=DeploymentTriggerConfig(min_accuracy=min_accuracy))
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )


@pipeline(enable_cache=False , settings={"docker":docker_settings})
def inference_pipeline(pipeline_name: str , pipeline_step_name: str):
    data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running = False
    )
    predictions = predictor(
        service=service,
        data=data,
    )
    return predictions

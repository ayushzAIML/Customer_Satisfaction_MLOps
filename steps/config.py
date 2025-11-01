# from zenml.config import BaseParameters

# class ModelNameConfig(BaseParameters):
#     """Model Configs"""

#     model_name: str = "LinearRegression"

from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """Model Configs"""
    model_name: str = "LinearRegression"


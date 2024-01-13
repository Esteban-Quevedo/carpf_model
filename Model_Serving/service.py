# -*- coding: utf-8 -*-

"""
predict_service.py: This script defines a BentoML service for predicting the power performance factor
using a trained model and Pydantic input data.

Author: Esteban Quevedo Pardo
"""

from bentoml.io import NumpyNdarray, JSON
from pydantic import BaseModel
import pandas as pd
import numpy as np
import bentoml

# Load the BentoML model
car_mod_runner = bentoml.sklearn.get("car_power_factor_model:latest").to_runner()

# Create a BentoML service
svc = bentoml.Service("car_power", runners=[car_mod_runner])


# Define input data model using Pydantic
class Customer(BaseModel):
    Manufacturer: int = 1
    Vehicle_type: int = 2
    Price_in_thousands: float = 21.5
    Engine_size: float = 1.8
    Horsepower: int = 140
    Wheelbase: float = 101.2
    Width: float = 67.3
    Length: float = 172.4
    Curb_weight: float = 2.639
    Fuel_capacity: float = 13.2
    Fuel_efficiency: int = 28


# Define BentoML API for prediction
@svc.api(input=JSON(pydantic_model=Customer), output=NumpyNdarray())
def predict(data: Customer) -> np.array:
    """
    Predict the power performance factor using the trained model.

    Args:
    - data (Customer): Input data in Pydantic model format.

    Returns:
    np.array: Predicted power performance factor.
    """
    # Convert Pydantic model to DataFrame
    df = pd.DataFrame(data.dict(), index=[0])

    # Run prediction using BentoML model runner
    result = car_mod_runner.predict.run(df)

    # Convert the result to NumPy array
    return np.array(result)

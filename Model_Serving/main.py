# -*- coding: utf-8 -*-

"""
Main.py: This script processes the car dataset, trains a linear regression model,
and saves the model using BentoML.

Author: Esteban Quevedo Pardo
"""

from sklearn.linear_model import LinearRegression
import pandas as pd
import bentoml


def manufacturer_replacement(dtf):
    """
    Replace manufacturer names with numerical values in the dataset.

    Args:
    - dtf (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: DataFrame with replaced values.
    """
    manu_vals = {"Manufacturer": {'Acura': 1, 'Audi': 2, 'BMW': 3, 'Buick': 4, 'Cadillac': 5, 'Chevrolet': 6,
                                  'Chrysler': 7, 'Dodge': 8, 'Ford': 9, 'Honda': 10, 'Hyundai': 11, 'Infiniti': 12,
                                  'Jaguar': 13, 'Jeep': 14, 'Lexus': 15, 'Lincoln': 16, 'Mitsubishi': 17, 'Mercury': 18,
                                  'Mercedes-B': 19, 'Nissan': 20, 'Oldsmobile': 21, 'Plymouth': 22, 'Pontiac': 23,
                                  'Porsche': 24, 'Saab': 25, 'Saturn': 26, 'Subaru': 27, 'Toyota': 28, 'Volkswagen': 29,
                                  'Volvo': 30}}
    dtf.replace(manu_vals, inplace=True)
    return dtf


def vehicle_type_replacement(dtf):
    """
    Replace vehicle types with numerical values in the dataset.

    Args:
    - dtf (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: DataFrame with replaced values.
    """
    type_vals = {"Vehicle_type": {'Car': 1, 'Passenger': 2}}
    dtf.replace(type_vals, inplace=True)
    return dtf


def get_dataset():
    """
    Read and return the car dataset.

    Returns:
    pd.DataFrame: Car dataset.
    """
    dtf = pd.read_csv('data/car_dataset.csv')
    return dtf


def process_data(dtf):
    """
    Process the car dataset by replacing values and filtering data.

    Args:
    - dtf (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: Processed DataFrame.
    """
    dtf = manufacturer_replacement(dtf)
    dtf = vehicle_type_replacement(dtf)
    dtf = dtf.query('Price_in_thousands > 0 & Price_in_thousands < 50')
    dtf = dtf.query('Engine_size > 0 or Engine_size < 8')
    dtf = dtf.query('Horsepower > 100')
    dtf = dtf.query('Curb_weight > 0.5')
    dtf = dtf.query('Fuel_efficiency > 10 & Fuel_efficiency < 40')
    dtf = dtf.query('Power_perf_factor > 25 & Power_perf_factor < 150')
    dtf = dtf.drop(['Sales_in_thousands', 'year_resale_value'], axis=1)
    return dtf


def create_model():
    """
    Create and return a linear regression model.

    Returns:
    sklearn.linear_model.LinearRegression: Linear regression model.
    """
    model = LinearRegression()
    return model


def train_and_save_model(model, trn_data, trn_target):
    """
    Train the linear regression model and save it using BentoML.

    Args:
    - model (sklearn.linear_model.LinearRegression): Linear regression model.
    - trn_data (pd.DataFrame): Training data.
    - trn_target (pd.Series): Target values for training.

    Returns:
    str: Path to the saved BentoML model.
    """
    model.fit(X=trn_data, y=trn_target)
    saved_model = bentoml.sklearn.save_model("car_power_factor_model", model)
    return saved_model


if __name__ == "__main__":
    # Load dataset
    df = get_dataset()

    # Process data
    training_data = process_data(df)

    # Separate target variable and features
    trn_target = training_data.pop('Power_perf_factor')
    trn_data = training_data

    # Create the model
    linear_model = create_model()

    # Train and save the model
    saved_model_path = train_and_save_model(linear_model, trn_data, trn_target)

    # Print a confirmation message that the model was saved and it's version.
    print(f"Model saved: {saved_model_path}")

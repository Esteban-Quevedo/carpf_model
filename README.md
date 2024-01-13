# Car Power Factor ML Model

## Objective

This project aims to create a machine learning model called Car Power and make predictions on the power factor
using [BentoML](https://docs.bentoml.com/en/latest/ "BentoML Documentation") (“a framework for building reliable,
scalable and cost-efficient AI applications.”) and a free dataset
called [car_dataset](https://www.kaggle.com/datasets/willstr/car-salescsv "Dataset Source").

## Content

### Understanding and Analyzing the Data

The first resource is a Jupyter Notebook that contains a code that provides a comprehensive analysis of the dataset,
including data loading, exploration, visualization, and preprocessing. The scatter plots are particularly useful for
understanding the relationships between different variables in the dataset. The data preprocessing steps aim to filter
the data based on certain conditions and remove unnecessary columns.

[Go to Dataset Analysis](https://github.com/Esteban-Quevedo/carpf_model/tree/main/Dataset_Analysis)

### Serving an ML Model using BentoML

The second resource is a folder that contains all the requirements to create and serve the Car power factor model
through BentoML. With the files required to create a predictive model, serve it locally (using a service python script
and Bento
built), and test its functionality on your machines.

Here you will find 4 files required for the lab execution and model serving:

* **requirements.txt:** Contains all the Python Packages required for the execution of the laboratory.
* **main.py:** This code processes the car dataset, trains a linear
  regression model, and saves the model using BentoML.
* **service.py:** This script defines a BentoML service for predicting
  the power performance factor using a trained model and Pydantic input data.
* **bentofile.yaml:** Configuration file for
  BentoML service.

[Go to Model Serving](https://github.com/Esteban-Quevedo/carpf_model/tree/main/Model_Serving)

## Documentation
- Official Documentation: [Document Page](https://eqpsolutions.com/blog/odoo-modules-documentation-1/product-certificates-module-1)

## Contacts
- Website: [Form Link](https://eqpsolutions.com/contactus)
- E-mail: [info@eqpsolutions.com](mailto:info@eqpsolutions.com)


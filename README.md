# Telco-Customer-Churn
==============================

Telco Customer Churn Prediction Project

To see the data dictionary, you could visit: https://www.kaggle.com/code/rahaffayez/telco-customer-churn/notebook

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── scores         <- Results from scoring model.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering)
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── make_dataset.py<- Script to prepare data
    │   │
    │   ├── train.py       <- Script to train models
    │   │                    
    │   ├── evaluate.py    <- Script to evaluate models using kpi's
    │   │
    │   └── predict.py     <- Script to use trained models to make predictions
    │
    └── LICENSE            <- License


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>